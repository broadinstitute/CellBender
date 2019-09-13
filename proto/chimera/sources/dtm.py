import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator

from boltons.cacheutils import cachedproperty

import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.distributions import TorchDistribution

import torch
from torch.distributions import constraints

from pyro_extras import CustomLogProbTerm, MixtureDistribution, logaddexp, get_log_prob_compl, \
    get_binomial_samples_sparse_counts
from fingerprint import SingleCellFingerprintDTM
from fsd import FSDCodec, SortByComponentWeights
from expr import GeneExpressionPrior
from importance_sampling import PosteriorImportanceSamplerInputs, PosteriorImportanceSampler
from stats import int_ndarray_mode, gamma_loc_scale_to_concentration_rate

import logging
from collections import defaultdict


class DropletTimeMachineModel(torch.nn.Module):

    def __init__(self,
                 init_params_dict: Dict[str, Union[int, float, bool]],
                 model_constraint_params_dict: Dict[str, Dict[str, Union[int, float]]],
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 gene_expression_prior: GeneExpressionPrior,
                 fsd_codec: FSDCodec,
                 guide_spec_dict: Dict[str, str],
                 device=torch.device('cuda'),
                 dtype=torch.float):
        super(DropletTimeMachineModel, self).__init__()

        self.model_constraint_params_dict = model_constraint_params_dict
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.gene_expression_prior = gene_expression_prior
        self.fsd_codec = fsd_codec
        self.guide_spec_dict = guide_spec_dict

        self.device = device
        self.dtype = dtype

        # hyperparameters
        self.fsd_gmm_num_components: int = init_params_dict['fsd.gmm_num_components']
        self.fsd_gmm_dirichlet_concentration: float = init_params_dict['fsd.gmm_dirichlet_concentration']
        self.fsd_gmm_init_xi_scale: float = init_params_dict['fsd.gmm_init_xi_scale']
        self.fsd_gmm_min_xi_scale: float = init_params_dict['fsd.gmm_min_xi_scale']
        self.fsd_gmm_max_xi_scale: Union[None, float] = init_params_dict['fsd.gmm_max_xi_scale']
        self.fsd_gmm_init_components_perplexity: float = init_params_dict['fsd.gmm_init_components_perplexity']
        self.fsd_gmm_min_weight_per_component: float = init_params_dict['fsd.gmm_min_weight_per_component']
        self.enable_fsd_w_dirichlet_reg: bool = init_params_dict['fsd.enable_fsd_w_dirichlet_reg']
        self.w_lo_dirichlet_reg_strength: float = init_params_dict['fsd.w_lo_dirichlet_reg_strength']
        self.w_hi_dirichlet_reg_strength: float = init_params_dict['fsd.w_hi_dirichlet_reg_strength']
        self.w_lo_dirichlet_concentration: float = init_params_dict['fsd.w_lo_dirichlet_concentration']
        self.w_hi_dirichlet_concentration: float = init_params_dict['fsd.w_hi_dirichlet_concentration']
        self.fsd_xi_posterior_min_scale: float = init_params_dict['fsd.xi_posterior_min_scale']
        self.n_particles_fingerprint_log_like: int = init_params_dict['model.n_particles_fingerprint_log_like']

        self.alpha_c_prior_a, self.alpha_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['global.chimera_alpha_c_prior_loc'],
            scale=init_params_dict['global.chimera_alpha_c_prior_scale'])

        self.beta_c_prior_a, self.beta_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['global.chimera_beta_c_prior_loc'],
            scale=init_params_dict['global.chimera_beta_c_prior_scale'])

        # empirical normalization factors
        self.mean_total_molecules_per_cell: float = np.mean(sc_fingerprint_dtm.total_obs_molecules_per_cell).item()
        self.mean_empirical_fsd_mu_hi: float = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi).item()

        # logging
        self._logger = logging.getLogger()
                
    def forward(self, _):
        raise NotImplementedError

    @cachedproperty
    def numpy_dtype(self):
        if self.dtype == torch.float16:
            return np.float16
        if self.dtype == torch.float32:
            return np.float32
        if self.dtype == torch.float64:
            return np.float64
        else:
            raise ValueError("Bad torch dtype -- allowed values are: float16, float32, float64")

    @cachedproperty
    def eps(self):
        return np.finfo(self.numpy_dtype).eps

    def model(self,
              data: Dict[str, torch.Tensor],
              posterior_sampling_mode: bool = False):
        """
        .. note:: in the variables, we use prefix ``n`` for batch index, ``k`` for mixture component index,
            ``r`` for family size, ``g`` for gene index, ``q`` for the dimensions of the encoded fsd repr,
            and ``j`` for fsd components (could be different for lo and hi components).
        """

        # input tensors
        fingerprint_tensor_nr = data['fingerprint_tensor']
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        cell_sampling_site_scale_factor_tensor_n = data['cell_sampling_site_scale_factor_tensor']
        downsampling_rate_tensor_n = data['downsampling_rate_tensor']
        empirical_fsd_mu_hi_tensor_n = data['empirical_fsd_mu_hi_tensor']
        empirical_mean_obs_expr_per_gene_tensor_n = data['empirical_mean_obs_expr_per_gene_tensor']
        gene_index_tensor_n = data['gene_index_tensor']
        cell_index_tensor_n = data['cell_index_tensor']
        total_obs_reads_per_cell_tensor_n = data['total_obs_reads_per_cell_tensor']
        total_obs_molecules_per_cell_tensor_n = data['total_obs_molecules_per_cell_tensor']
        cell_features_tensor_nf = data['cell_features_tensor']

        # sizes
        mb_size = fingerprint_tensor_nr.shape[0]
        batch_shape = torch.Size([mb_size])
        max_family_size = fingerprint_tensor_nr.shape[1]

        # register the external modules
        pyro.module("fsd_codec", self.fsd_codec)
        pyro.module("gene_expression_prior", self.gene_expression_prior)
        
        # GMM prior for family size distribution parameters
        fsd_xi_prior_locs_kq = pyro.param(
            "fsd_xi_prior_locs_kq",
            self.fsd_codec.init_fsd_xi_loc_prior +
            self.fsd_gmm_init_components_perplexity * torch.randn(
                (self.fsd_gmm_num_components, self.fsd_codec.total_fsd_params),
                dtype=self.dtype, device=self.device))

        if self.fsd_gmm_max_xi_scale is None:
            fsd_xi_prior_scales_constraint = constraints.greater_than(self.fsd_gmm_min_xi_scale)
        else:
            fsd_xi_prior_scales_constraint = constraints.interval(
                self.fsd_gmm_min_xi_scale, self.fsd_gmm_max_xi_scale)
        fsd_xi_prior_scales_kq = pyro.param(
            "fsd_xi_prior_scales_kq",
            self.fsd_gmm_init_xi_scale * torch.ones(
                (self.fsd_gmm_num_components, self.fsd_codec.total_fsd_params),
                dtype=self.dtype, device=self.device),
            constraint=fsd_xi_prior_scales_constraint)

        # chimera hyperparameters
        alpha_c_concentration_scalar = torch.tensor(
            self.alpha_c_prior_a, device=self.device, dtype=self.dtype)
        alpha_c_rate_scalar = torch.tensor(
            self.alpha_c_prior_b, device=self.device, dtype=self.dtype)
        beta_c_concentration_scalar = torch.tensor(
            self.beta_c_prior_a, device=self.device, dtype=self.dtype)
        beta_c_rate_scalar = torch.tensor(
            self.beta_c_prior_b, device=self.device, dtype=self.dtype)

        # useful auxiliary quantities
        family_size_vector_obs_r = torch.arange(
            1, max_family_size + 1, device=self.device, dtype=self.dtype)
        family_size_vector_full_r = torch.arange(
            0, max_family_size + 1, device=self.device, dtype=self.dtype)
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)

        # fsd xi prior distribution
        fsd_xi_prior_dist = self._get_fsd_xi_prior_dist(
            fsd_xi_prior_locs_kq,
            fsd_xi_prior_scales_kq)

        # sample chimera parameters
        alpha_c = pyro.sample(
            "alpha_c",
            dist.Gamma(concentration=alpha_c_concentration_scalar, rate=alpha_c_rate_scalar))
        beta_c = pyro.sample(
            "beta_c",
            dist.Gamma(concentration=beta_c_concentration_scalar, rate=beta_c_rate_scalar))

        with pyro.plate("collapsed_gene_cell", size=mb_size):

            with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
                # sample gene family size distribution parameters
                fsd_xi_nq = pyro.sample("fsd_xi_nq", fsd_xi_prior_dist)

            # empirical droplet efficiency
            eta_n = total_obs_molecules_per_cell_tensor_n / self.mean_total_molecules_per_cell

            # transform fsd xi to the constrained space
            fsd_params_dict = self.fsd_codec.decode(fsd_xi_nq)

            # get chimeric and real family size distributions
            fsd_lo_dist, fsd_hi_dist = self.fsd_codec.get_fsd_components(
                fsd_params_dict,
                downsampling_rate_tensor=downsampling_rate_tensor_n)

            # get e_hi prior parameters (per cell)
            beta_loc_nr, beta_scale_nr = self.gene_expression_prior.forward(data)

            # sample e_hi prior parameters
            with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
                # sample beta parameters
                beta_nr = pyro.sample(
                    "beta_nr",
                    dist.Normal(loc=beta_loc_nr, scale=beta_scale_nr).to_event(1))

            # calculate ZINB parameters
            log_eta_n = eta_n.log()
            log_mu_e_hi_n = beta_nr[:, 0] + beta_nr[:, 1] * log_eta_n
            log_phi_e_hi_n = beta_nr[:, 2]
            logit_p_zero_e_hi_n = beta_nr[:, 3]

            mu_e_hi_n = log_mu_e_hi_n.exp()
            phi_e_hi_n = log_phi_e_hi_n.exp()

            # extract required quantities from the distributions
            mu_fsd_lo_n = fsd_lo_dist.mean.squeeze(-1)
            mu_fsd_hi_n = fsd_hi_dist.mean.squeeze(-1)
            log_p_unobs_lo_n = fsd_lo_dist.log_prob(zero).squeeze(-1)
            log_p_unobs_hi_n = fsd_hi_dist.log_prob(zero).squeeze(-1)
            log_p_obs_lo_n = get_log_prob_compl(log_p_unobs_lo_n)
            log_p_obs_hi_n = get_log_prob_compl(log_p_unobs_hi_n)
            p_obs_lo_n = log_p_obs_lo_n.exp()
            p_obs_hi_n = log_p_obs_hi_n.exp()

            # localization and/or calculation of required variables for pickup by locals() -- see below
            phi_fsd_lo_comps_nj = fsd_params_dict['phi_lo']
            phi_fsd_hi_comps_nj = fsd_params_dict['phi_hi']
            mu_fsd_lo_comps_nj = fsd_params_dict['mu_lo']
            mu_fsd_hi_comps_nj = fsd_params_dict['mu_hi']
            w_fsd_lo_comps_nj = fsd_params_dict['w_lo']
            w_fsd_hi_comps_nj = fsd_params_dict['w_hi']

            # for regularization
            p_obs_lo_to_p_obs_hi_ratio_n = p_obs_lo_n / p_obs_hi_n
            mu_fsd_lo_to_mu_fsd_hi_ratio_n = mu_fsd_lo_n / mu_fsd_hi_n
            mu_fsd_lo_comps_to_mu_empirical_ratio_nj = mu_fsd_lo_comps_nj / self.mean_empirical_fsd_mu_hi
            mu_fsd_hi_comps_to_mu_empirical_ratio_nj = mu_fsd_hi_comps_nj / self.mean_empirical_fsd_mu_hi

            # observation probability for each component of the distribution
            alpha_fsd_lo_comps_nj = (self.eps + phi_fsd_lo_comps_nj).reciprocal()
            log_p_unobs_lo_comps_nj = alpha_fsd_lo_comps_nj * (
                    alpha_fsd_lo_comps_nj.log() - (alpha_fsd_lo_comps_nj + mu_fsd_lo_comps_nj).log())
            p_obs_lo_comps_nj = get_log_prob_compl(log_p_unobs_lo_comps_nj).exp()
            alpha_fsd_hi_comps_nj = (self.eps + phi_fsd_hi_comps_nj).reciprocal()
            log_p_unobs_hi_comps_nj = alpha_fsd_hi_comps_nj * (
                    alpha_fsd_hi_comps_nj.log() - (alpha_fsd_hi_comps_nj + mu_fsd_hi_comps_nj).log())
            p_obs_hi_comps_nj = get_log_prob_compl(log_p_unobs_hi_comps_nj).exp()

            # calculate p_lo and p_hi on all observable family sizes
            log_prob_fsd_lo_full_nr = fsd_lo_dist.log_prob(family_size_vector_full_r)
            log_prob_fsd_hi_full_nr = fsd_hi_dist.log_prob(family_size_vector_full_r)
            log_prob_fsd_lo_obs_nr = log_prob_fsd_lo_full_nr[..., 1:]
            log_prob_fsd_hi_obs_nr = log_prob_fsd_hi_full_nr[..., 1:]

            # total observed molecules per cell
            e_obs_n = fingerprint_tensor_nr.sum(-1)

            # calculate the (poisson) rate of chimeric molecule formation
            mu_e_lo_n = self._get_mu_e_lo_n(
                alpha_c=alpha_c,
                beta_c=beta_c,
                eta_n=eta_n,
                mean_empirical_fsd_mu_hi=self.mean_empirical_fsd_mu_hi,
                empirical_mean_obs_expr_per_gene_tensor_n=empirical_mean_obs_expr_per_gene_tensor_n,
                p_obs_lo_n=p_obs_lo_n,
                p_obs_hi_n=p_obs_hi_n,
                mu_fsd_hi_n=mu_fsd_hi_n,
                downsampling_rate_tensor_n=downsampling_rate_tensor_n)

            if posterior_sampling_mode:

                # just return the calculated auxiliary tensors
                return locals()

            else:

                # sample the fingerprint
                self._sample_fingerprint(
                    batch_shape=batch_shape,
                    cell_sampling_site_scale_factor_tensor_n=cell_sampling_site_scale_factor_tensor_n,
                    fingerprint_tensor_nr=fingerprint_tensor_nr,
                    log_prob_fsd_lo_obs_nr=log_prob_fsd_lo_obs_nr,
                    log_prob_fsd_hi_obs_nr=log_prob_fsd_hi_obs_nr,
                    mu_e_lo_n=mu_e_lo_n,
                    mu_e_hi_n=mu_e_hi_n,
                    phi_e_hi_n=phi_e_hi_n,
                    logit_p_zero_e_hi_n=logit_p_zero_e_hi_n,
                    n_particles=self.n_particles_fingerprint_log_like)

                # sample fsd sparsity regularization
                if self.enable_fsd_w_dirichlet_reg:
                    self._sample_fsd_weight_sparsity_regularization(
                        w_lo_dirichlet_reg_strength=self.w_lo_dirichlet_reg_strength,
                        w_hi_dirichlet_reg_strength=self.w_hi_dirichlet_reg_strength,
                        w_lo_dirichlet_concentration=self.w_lo_dirichlet_concentration,
                        w_hi_dirichlet_concentration=self.w_hi_dirichlet_concentration,
                        n_fsd_lo_comps=self.fsd_codec.n_fsd_lo_comps,
                        n_fsd_hi_comps=self.fsd_codec.n_fsd_hi_comps,
                        w_fsd_lo_comps_nj=w_fsd_lo_comps_nj,
                        w_fsd_hi_comps_nj=w_fsd_hi_comps_nj,
                        gene_sampling_site_scale_factor_tensor_n=gene_sampling_site_scale_factor_tensor_n)

                # sample (soft) constraints
                self._sample_gene_plate_soft_constraints(
                    model_constraint_params_dict=self.model_constraint_params_dict,
                    model_vars_dict=locals(),
                    gene_sampling_site_scale_factor_tensor_n=gene_sampling_site_scale_factor_tensor_n,
                    batch_shape=batch_shape)

    @staticmethod
    def _sample_fingerprint(batch_shape: torch.Size,
                            cell_sampling_site_scale_factor_tensor_n: torch.Tensor,
                            fingerprint_tensor_nr: torch.Tensor,
                            log_prob_fsd_lo_obs_nr: torch.Tensor,
                            log_prob_fsd_hi_obs_nr: torch.Tensor,
                            mu_e_lo_n: torch.Tensor,
                            mu_e_hi_n: torch.Tensor,
                            phi_e_hi_n: torch.Tensor,
                            logit_p_zero_e_hi_n: torch.Tensor,
                            n_particles: int):

        # calculate the fingerprint log likelihood
        fingerprint_log_likelihood_n = DropletTimeMachineModel._get_fingerprint_log_likelihood_monte_carlo(
            fingerprint_tensor_nr=fingerprint_tensor_nr,
            log_prob_fsd_lo_obs_nr=log_prob_fsd_lo_obs_nr,
            log_prob_fsd_hi_obs_nr=log_prob_fsd_hi_obs_nr,
            mu_e_lo_n=mu_e_lo_n,
            mu_e_hi_n=mu_e_hi_n,
            phi_e_hi_n=phi_e_hi_n,
            logit_p_zero_e_hi_n=logit_p_zero_e_hi_n,
            n_particles=n_particles)

        # sample
        with poutine.scale(scale=cell_sampling_site_scale_factor_tensor_n):
            pyro.sample("fingerprint_and_expression_observation",
                        CustomLogProbTerm(
                            custom_log_prob=fingerprint_log_likelihood_n,
                            batch_shape=batch_shape,
                            event_shape=torch.Size([])),
                        obs=torch.zeros_like(fingerprint_log_likelihood_n))

    @staticmethod
    def _get_mu_e_lo_n(
            alpha_c: torch.Tensor,
            beta_c: torch.Tensor,
            eta_n: torch.Tensor,
            mean_empirical_fsd_mu_hi: float,
            empirical_mean_obs_expr_per_gene_tensor_n: torch.Tensor,
            p_obs_lo_n: torch.Tensor,
            p_obs_hi_n: torch.Tensor,
            mu_fsd_hi_n: torch.Tensor,
            downsampling_rate_tensor_n: torch.Tensor) -> torch.Tensor:
        """Calculates the Poisson rate of chimeric molecule formation

        :param alpha_c: chimera formation coefficient (cell-size prefactor)
        :param beta_c: chimera formation coefficient (constant piece)
        :param eta_n: (relative) cell size scale factor
        :param mean_empirical_fsd_mu_hi: empirical dataset-wide mean reads-per-molecule
        :param empirical_mean_obs_expr_per_gene_tensor_n: empirical mean gene expression per cell
        :param p_obs_lo_n: probability of observing a chimeric molecule
        :param p_obs_hi_n: probability of observing a real molecule
        :param mu_fsd_hi_n: mean family size per real molecule
        :param downsampling_rate_tensor_n: downsampling scale-factor of raw counts (applicable
            only if downsampling regularization is used -- deprecated)
        :return: Poisson rate of chimeric molecule formation
        """
        estimated_mean_e_hi_n = empirical_mean_obs_expr_per_gene_tensor_n / p_obs_hi_n
        scaled_mu_fsd_hi_n = mu_fsd_hi_n / (downsampling_rate_tensor_n * mean_empirical_fsd_mu_hi)
        scaled_total_fragments_n = estimated_mean_e_hi_n * scaled_mu_fsd_hi_n
        mu_e_lo_n = (alpha_c * eta_n + beta_c) * scaled_total_fragments_n
        return mu_e_lo_n

    @staticmethod
    def _get_fingerprint_log_likelihood_monte_carlo(fingerprint_tensor_nr: torch.Tensor,
                                                    log_prob_fsd_lo_obs_nr: torch.Tensor,
                                                    log_prob_fsd_hi_obs_nr: torch.Tensor,
                                                    mu_e_lo_n: torch.Tensor,
                                                    mu_e_hi_n: torch.Tensor,
                                                    phi_e_hi_n: torch.Tensor,
                                                    logit_p_zero_e_hi_n: torch.Tensor,
                                                    n_particles: int) -> torch.Tensor:
        """Calculates the approximate fingerprint log likelihood by marginalizing the zero-inflated
        Gamma (ZIG) prior distribution of real gene expression rate via Monte-Carlo sampling.

        .. note:: Importantly, the samples drawn of the ZIG distribution must be differentiable
            (e.g. re-parametrized) w.r.t. to its parameters.

        .. note:: For the purposes of model fitting, the overall data-dependent normalization factor
            of the log likelihood is immaterial and we drop it.

        :param n_particles: number of MC samples to draw for marginalizing the ZIG prior for :math:`e^>`.
        :return: fingerprint log likelihood
        """
        # pre-compute useful tensors
        p_lo_obs_nr = log_prob_fsd_lo_obs_nr.exp()
        total_obs_rate_lo_n = mu_e_lo_n * p_lo_obs_nr.sum(-1)
        log_rate_e_lo_nr = mu_e_lo_n.log().unsqueeze(-1) + log_prob_fsd_lo_obs_nr
        fingerprint_log_norm_factor_n = (fingerprint_tensor_nr + 1).lgamma().sum(-1)

        p_hi_obs_nr = log_prob_fsd_hi_obs_nr.exp()
        total_obs_rate_hi_n = mu_e_hi_n * p_hi_obs_nr.sum(-1)
        log_rate_e_hi_nr = mu_e_hi_n.log().unsqueeze(-1) + log_prob_fsd_hi_obs_nr

        log_p_zero_e_hi_n = torch.nn.functional.logsigmoid(logit_p_zero_e_hi_n)
        log_p_nonzero_e_hi_n = get_log_prob_compl(log_p_zero_e_hi_n)

        # zero-inflated contribution

        log_poisson_zero_e_hi_contrib_n = (
                (fingerprint_tensor_nr * log_rate_e_lo_nr).sum(-1)
                - total_obs_rate_lo_n
                - fingerprint_log_norm_factor_n)  # data-dependent norm factor can be dropped

        # non-zero-inflated contribution

        # step 1. draw re-parametrized Gamma particles
        alpha_e_hi_n = phi_e_hi_n.reciprocal()
        omega_mn = dist.Gamma(concentration=alpha_e_hi_n, rate=alpha_e_hi_n).rsample((n_particles,))

        # step 2. calculate the conditional log likelihood for each of the Gamma particles
        log_rate_combined_mnr = logaddexp(
            log_rate_e_lo_nr,
            log_rate_e_hi_nr + omega_mn.log().unsqueeze(-1))
        log_poisson_nonzero_e_hi_contrib_mn = (
            (fingerprint_tensor_nr * log_rate_combined_mnr).sum(-1)
            - (total_obs_rate_lo_n + total_obs_rate_hi_n * omega_mn)
            - fingerprint_log_norm_factor_n)  # data-dependent norm factor can be dropped

        # step 3. average over the Gamma particles
        log_poisson_nonzero_e_hi_contrib_n = log_poisson_nonzero_e_hi_contrib_mn.logsumexp(0) - np.log(n_particles)

        # put the zero-inflated and non-zero-inflated contributions together
        log_like_n = logaddexp(
            log_p_zero_e_hi_n + log_poisson_zero_e_hi_contrib_n,
            log_p_nonzero_e_hi_n + log_poisson_nonzero_e_hi_contrib_n)

        return log_like_n

    def _get_fsd_xi_prior_dist(self,
                               fsd_xi_prior_locs_kq: torch.Tensor,
                               fsd_xi_prior_scales_kq: torch.Tensor) -> TorchDistribution:
        """Calculates the prior distribution for :math:`\\xi`, which is a marginalized Gaussian mixture.

        :param fsd_xi_prior_locs_kq: location of Gaussian components
        :param fsd_xi_prior_scales_kq: scale of Gaussian components
        :return: a TorchDistribution
        """
        if self.fsd_gmm_num_components > 1:
            # generate the marginalized GMM distribution w/ Dirichlet prior over the weights
            fsd_xi_prior_weights_k = pyro.sample(
                "fsd_xi_prior_weights_k",
                dist.Dirichlet(
                    self.fsd_gmm_dirichlet_concentration *
                    torch.ones((self.fsd_gmm_num_components,), dtype=self.dtype, device=self.device)))
            fsd_xi_prior_log_weights_k = fsd_xi_prior_weights_k.log()
            fsd_xi_prior_log_weights_tuple = tuple(
                fsd_xi_prior_log_weights_k[k]
                for k in range(self.fsd_gmm_num_components))
            fsd_xi_prior_components_tuple = tuple(
                dist.Normal(fsd_xi_prior_locs_kq[k, :], fsd_xi_prior_scales_kq[k, :]).to_event(1)
                for k in range(self.fsd_gmm_num_components))
            fsd_xi_prior_dist = MixtureDistribution(
                fsd_xi_prior_log_weights_tuple,
                fsd_xi_prior_components_tuple)

        else:
            fsd_xi_prior_dist = dist.Normal(
                fsd_xi_prior_locs_kq[0, :],
                fsd_xi_prior_scales_kq[0, :]).to_event(1)

        return fsd_xi_prior_dist

    @staticmethod
    def _sample_gene_plate_soft_constraints(
            model_constraint_params_dict: Dict[str, Dict[str, float]],
            model_vars_dict: Dict[str, torch.Tensor],
            gene_sampling_site_scale_factor_tensor_n: torch.Tensor,
            batch_shape: torch.Size):
        """Imposes constraints on gene-dependent quantities by adding penalties to the model free energy.

        :param model_constraint_params_dict: constraint descriptor
        :param model_vars_dict: a dictionary that includes references to all to-be-constrained quantities
        :param gene_sampling_site_scale_factor_tensor_n: scale factor (for stratified mini-batching)
        :param batch_shape:
        :return:
        """
        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            for var_name, var_constraint_params in model_constraint_params_dict.items():
                var = model_vars_dict[var_name]
                if 'lower_bound_value' in var_constraint_params:
                    value = var_constraint_params['lower_bound_value']
                    width = var_constraint_params['lower_bound_width']
                    exponent = var_constraint_params['lower_bound_exponent']
                    strength = var_constraint_params['lower_bound_strength']
                    if isinstance(value, str):
                        value = model_vars_dict[value]
                    activity = torch.clamp(value + width - var, min=0.) / width
                    constraint_log_prob = - strength * activity.pow(exponent)
                    for _ in range(len(var.shape) - 1):
                        constraint_log_prob = constraint_log_prob.sum(-1)
                    pyro.sample(
                        var_name + "_lower_bound_constraint",
                        CustomLogProbTerm(constraint_log_prob,
                                          batch_shape=batch_shape,
                                          event_shape=torch.Size([])),
                        obs=torch.zeros_like(constraint_log_prob))

                if 'upper_bound_value' in var_constraint_params:
                    value = var_constraint_params['upper_bound_value']
                    width = var_constraint_params['upper_bound_width']
                    exponent = var_constraint_params['upper_bound_exponent']
                    strength = var_constraint_params['upper_bound_strength']
                    if isinstance(value, str):
                        value = model_vars_dict[value]
                    activity = torch.clamp(var - value + width, min=0.) / width
                    constraint_log_prob = - strength * activity.pow(exponent)
                    for _ in range(len(var.shape) - 1):
                        constraint_log_prob = constraint_log_prob.sum(-1)
                    pyro.sample(
                        var_name + "_upper_bound_constraint",
                        CustomLogProbTerm(constraint_log_prob,
                                          batch_shape=batch_shape,
                                          event_shape=torch.Size([])),
                        obs=torch.zeros_like(constraint_log_prob))

                if 'pin_value' in var_constraint_params:
                    value = var_constraint_params['pin_value']
                    exponent = var_constraint_params['pin_exponent']
                    strength = var_constraint_params['pin_strength']
                    if isinstance(value, str):
                        value = model_vars_dict[value]
                    activity = (var - value).abs()
                    constraint_log_prob = - strength * activity.pow(exponent)
                    for _ in range(len(var.shape) - 1):
                        constraint_log_prob = constraint_log_prob.sum(-1)
                    pyro.sample(
                        var_name + "_pin_value_constraint",
                        CustomLogProbTerm(constraint_log_prob,
                                          batch_shape=batch_shape,
                                          event_shape=torch.Size([])),
                        obs=torch.zeros_like(constraint_log_prob))

    @staticmethod
    def _sample_fsd_weight_sparsity_regularization(
            w_lo_dirichlet_reg_strength: float,
            w_hi_dirichlet_reg_strength: float,
            w_lo_dirichlet_concentration: float,
            w_hi_dirichlet_concentration: float,
            n_fsd_lo_comps: int,
            n_fsd_hi_comps: int,
            w_fsd_lo_comps_nj: torch.Tensor,
            w_fsd_hi_comps_nj: torch.Tensor,
            gene_sampling_site_scale_factor_tensor_n: torch.Tensor):
        with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
            if n_fsd_lo_comps > 1:
                with poutine.scale(scale=w_lo_dirichlet_reg_strength):
                    pyro.sample(
                        "w_lo_dirichlet_reg",
                        dist.Dirichlet(w_lo_dirichlet_concentration * torch.ones_like(w_fsd_lo_comps_nj)),
                        obs=w_fsd_lo_comps_nj)
            if n_fsd_hi_comps > 1:
                with poutine.scale(scale=w_hi_dirichlet_reg_strength):
                    pyro.sample(
                        "w_hi_dirichlet_reg",
                        dist.Dirichlet(w_hi_dirichlet_concentration * torch.ones_like(w_fsd_hi_comps_nj)),
                        obs=w_fsd_hi_comps_nj)

    def guide(self,
              data: Dict[str, torch.Tensor],
              posterior_sampling_mode: bool = False):

        # input tensors
        fingerprint_tensor_nr = data['fingerprint_tensor']
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        gene_index_tensor_n = data['gene_index_tensor']

        # sizes
        mb_size = fingerprint_tensor_nr.shape[0]

        # register the external modules
        pyro.module("fsd_codec", self.fsd_codec)
        pyro.module("gene_expression_prior", self.gene_expression_prior)

        # fsd xi gmm
        if self.fsd_gmm_num_components > 1:
            # MAP estimate of GMM fsd prior weights
            fsd_xi_prior_weights_map_k = pyro.param(
                "fsd_xi_prior_weights_map_k",
                torch.ones((self.fsd_gmm_num_components,),
                           device=self.device, dtype=self.dtype) / self.fsd_gmm_num_components,
                constraint=constraints.simplex)
            pyro.sample(
                "fsd_xi_prior_weights_k",
                dist.Delta(
                    self.fsd_gmm_min_weight_per_component
                    + (1 - self.fsd_gmm_num_components * self.fsd_gmm_min_weight_per_component)
                    * fsd_xi_prior_weights_map_k))

        # point estimate for fsd xi (per gene)
        fsd_xi_posterior_loc_gq = pyro.param(
            "fsd_xi_posterior_loc_gq",
            self.fsd_codec.get_sorted_fsd_xi(self.fsd_codec.init_fsd_xi_loc_posterior))

        # point estimate for chimera parameters
        alpha_c_posterior_loc = pyro.param(
            "alpha_c_posterior_loc",
            torch.tensor(self.alpha_c_prior_a / self.alpha_c_prior_b, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        beta_c_posterior_loc = pyro.param(
            "beta_c_posterior_loc",
            torch.tensor(self.beta_c_prior_a / self.beta_c_prior_b, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        pyro.sample("alpha_c", dist.Delta(v=alpha_c_posterior_loc))
        pyro.sample("beta_c", dist.Delta(v=beta_c_posterior_loc))

        # base posterior distribution for fsd xi
        if self.guide_spec_dict['fsd_xi'] == 'map':

            fsd_xi_posterior_base_dist = dist.Delta(
                v=fsd_xi_posterior_loc_gq[gene_index_tensor_n, :]).to_event(1)

        elif self.guide_spec_dict['fsd_xi'] == 'gaussian':

            # scale for fsd xi (per gene)
            fsd_xi_posterior_scale_gq = pyro.param(
                "fsd_xi_posterior_scale_gq",
                self.fsd_gmm_init_xi_scale * torch.ones(
                    (self.sc_fingerprint_dtm.n_genes,
                     self.fsd_codec.total_fsd_params), device=self.device, dtype=self.dtype),
                constraint=constraints.greater_than(self.fsd_xi_posterior_min_scale))
            fsd_xi_posterior_base_dist = dist.Normal(
                loc=fsd_xi_posterior_loc_gq[gene_index_tensor_n, :],
                scale=fsd_xi_posterior_scale_gq[gene_index_tensor_n, :]).to_event(1)

        else:
            raise Exception("Unknown guide specification for fsd_xi: allowed values are 'map' and 'gaussian'")

        # apply a pseudo-bijective transformation to sort xi by component weights
        fsd_xi_sort_trans = SortByComponentWeights(self.fsd_codec)
        fsd_xi_posterior_dist = dist.TransformedDistribution(
            fsd_xi_posterior_base_dist, [fsd_xi_sort_trans])
        
        # gene expression prior guide
        self.gene_expression_prior.guide(data)

        with pyro.plate("collapsed_gene_cell", size=mb_size):

            with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
                pyro.sample("fsd_xi_nq", fsd_xi_posterior_dist)

    # TODO: rewrite using poutine and avoid code repetition
    @torch.no_grad()
    def get_active_constraints_on_genes(self) -> Dict:
        # TODO grab variables from the model
        raise NotImplementedError

        # model_vars_dict = ...
        active_constraints_dict = defaultdict(dict)
        for var_name, var_constraint_params in self.model_constraint_params_dict.items():
            var = model_vars_dict[var_name]
            if 'lower_bound_value' in var_constraint_params:
                value = var_constraint_params['lower_bound_value']
                width = var_constraint_params['lower_bound_width']
                if isinstance(value, str):
                    value = model_vars_dict[value]
                activity = torch.clamp(value + width - var, min=0.)
                for _ in range(len(var.shape) - 1):
                    activity = activity.sum(-1)
                nnz_activity = torch.nonzero(activity).cpu().numpy().flatten()
                if nnz_activity.size > 0:
                    active_constraints_dict[var_name]['lower_bound'] = set(nnz_activity.tolist())

            if 'upper_bound_value' in var_constraint_params:
                value = var_constraint_params['upper_bound_value']
                width = var_constraint_params['upper_bound_width']
                if isinstance(value, str):
                    value = model_vars_dict[value]
                activity = torch.clamp(var - value + width, min=0.)
                for _ in range(len(var.shape) - 1):
                    activity = activity.sum(-1)
                nnz_activity = torch.nonzero(activity).cpu().numpy().flatten()
                if nnz_activity.size > 0:
                    active_constraints_dict[var_name]['upper_bound'] = set(nnz_activity.tolist())

        return dict(active_constraints_dict)


class PosteriorGeneExpressionSampler(object):
    """Calculates the real gene expression from a trained model using importance sampling"""
    def __init__(self,
                 dtm_model: DropletTimeMachineModel,
                 device: torch.device,
                 dtype: torch.dtype):
        """Initializer.

        :param dtm_model: an instance of ``DropletTimeMachineModel``
        :param device: torch device
        :param dtype: torch dtype
        """
        self.dtm_model = dtm_model
        self.device = device
        self.dtype = dtype

    def _sharded_cell_index_generator(
            self,
            gene_index: int,
            cell_shard_size: int,
            only_expressing_cells: bool) -> Generator[List[int], None, None]:
        sc_fingerprint_dtm = self.dtm_model.sc_fingerprint_dtm
        if only_expressing_cells:
            included_cell_indices = sc_fingerprint_dtm.get_expressing_cell_indices(gene_index)
        else:
            included_cell_indices = np.arange(0, sc_fingerprint_dtm.n_cells).tolist()
        n_included_cells = len(included_cell_indices)

        for i_cell_shard in range(n_included_cells // cell_shard_size + 1):
            i_included_cell_begin = min(i_cell_shard * cell_shard_size, n_included_cells)
            i_included_cell_end = min((i_cell_shard + 1) * cell_shard_size, n_included_cells)
            if i_included_cell_begin == i_included_cell_end:
                break
            yield included_cell_indices[i_included_cell_begin:i_included_cell_end]

    @torch.no_grad()
    def _get_trained_model_context(self, minibatch_data: Dict[str, torch.Tensor]) \
            -> Dict[str, Any]:
        """Samples the posterior on a given minibatch, replays it on the model, and returns a
        dictionary of intermediate tensors that appear in the model evaluated at the posterior
        samples."""
        guide_trace = poutine.trace(self.dtm_model.guide).get_trace(
            minibatch_data, posterior_sampling_mode=True)
        trained_model = poutine.replay(self.dtm_model.model, trace=guide_trace)
        trained_model_trace = poutine.trace(trained_model).get_trace(
            minibatch_data, posterior_sampling_mode=True)
        return trained_model_trace.nodes["_RETURN"]["value"]

    @torch.no_grad()
    def _generate_omega_importance_sampler_inputs(
            self,
            gene_index: int,
            cell_index_list: List[int],
            n_particles_cell: int,
            run_mode: str) -> Tuple[PosteriorImportanceSamplerInputs, Dict[str, Any], Dict[str, Any]]:
        """Generates the required inputs for ``PosteriorImportanceSampler`` to calculate the mean
        and variance of gene expression, assuming no zero-inflation of the prior for :math:`\\mu^>`.

        .. note:: According to the model, the prior for :math:`\\mu^<`, the Poisson rate for chimeric
            molecules, is deterministic; as such, it is directly picked up from the trained model
            context ``trained_model_context["mu_e_lo_n"]`` and no marginalization is necessary.

            The prior :math:`\\mu^>`, the Poisson rate for real molecules, however, is a zero-inflated
            Gamma and must be marginalized. This method generates importance sampling inputs for
            marginalizing :math:`\\mu^>` and calculing the first and second moments of of gene expression,
            for a non-zero-inflated Gamma. We parametrize :math:`\\mu^>` as follows:

            .. math::

                \\mu^> | no-zero-inflation = \\mathbb{E}[\\mu^>] \\omega^>,

                \\omega^> \\sim \\mathrm{Gamma}(\\alpha^>, \\alpha^>),

                \\alpha^> = 1 / \\phi^>,

            where :math:`\\phi^>` is the prior over-dipersion of expression and :math:`\\mathbb{E}[\\mu^>]`
            is the prior mean of the expression.

        :param gene_index: index of the gene in the datastore
        :param run_mode: choices include ``only_observed`` and ``full``. In the only_observed mode,
            we only calculate how many of the observed molecules are real (as opposed to background/chimeric).
            In the ``full`` mode, we also include currently unobserved molecules; that is, we estimate
            how many real molecules we expect to observe in the limit of infinite sequencing depth.
        :param cell_index_list: list of cell indices to analyse
        :param n_particles_cell: how many repeats per cell
        :return: a tuple that matches the *args signature of ``PosteriorImportanceSampler.__init__``, and
            the trained model context.
        """
        assert run_mode in {"only_observed", "full"}
        minibatch_data = self.dtm_model.sc_fingerprint_dtm.generate_single_gene_minibatch_data(
            gene_index=gene_index,
            cell_index_list=cell_index_list,
            n_particles_cell=n_particles_cell)
        trained_model_context = self._get_trained_model_context(minibatch_data)

        # localize required auxiliary quantities from the trained model context
        fingerprint_tensor_nr = minibatch_data["fingerprint_tensor"]
        mu_e_hi_n: torch.Tensor = trained_model_context["mu_e_hi_n"]
        phi_e_hi_n: torch.Tensor = trained_model_context["phi_e_hi_n"]
        logit_p_zero_e_hi_n: torch.Tensor = trained_model_context["logit_p_zero_e_hi_n"]
        mu_e_lo_n: torch.Tensor = trained_model_context["mu_e_lo_n"]
        log_prob_fsd_lo_full_nr: torch.Tensor = trained_model_context["log_prob_fsd_lo_full_nr"]
        log_prob_fsd_hi_full_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_full_nr"]
        log_prob_fsd_lo_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_lo_obs_nr"]
        log_prob_fsd_hi_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_obs_nr"]

        # calculate additional auxiliary quantities
        alpha_e_hi_n = phi_e_hi_n.reciprocal()
        log_p_zero_e_hi_n = torch.nn.functional.logsigmoid(logit_p_zero_e_hi_n)
        log_p_nonzero_e_hi_n = get_log_prob_compl(log_p_zero_e_hi_n)
        p_nonzero_e_hi_n = log_p_nonzero_e_hi_n.exp()

        e_obs_n = fingerprint_tensor_nr.sum(-1)
        log_fingerprint_tensor_nr = fingerprint_tensor_nr.log()
        fingerprint_log_norm_factor_n = (fingerprint_tensor_nr + 1).lgamma().sum(-1)

        log_prob_unobs_lo_n = log_prob_fsd_lo_full_nr[..., 0]
        log_prob_unobs_hi_n = log_prob_fsd_hi_full_nr[..., 0]
        log_prob_obs_lo_n = get_log_prob_compl(log_prob_unobs_lo_n)
        log_prob_obs_hi_n = get_log_prob_compl(log_prob_unobs_hi_n)
        prob_obs_lo_n = log_prob_obs_lo_n.exp()
        prob_obs_hi_n = log_prob_obs_hi_n.exp()
        log_mu_e_lo_n = mu_e_lo_n.log()
        log_mu_e_hi_n = mu_e_hi_n.log()
        log_rate_e_lo_nr = log_mu_e_lo_n.unsqueeze(-1) + log_prob_fsd_lo_obs_nr
        log_rate_e_hi_nr = log_mu_e_hi_n.unsqueeze(-1) + log_prob_fsd_hi_obs_nr

        def fingerprint_log_like_function(omega_mn: torch.Tensor) -> torch.Tensor:
            """Calculates the log likelihood of the fingerprint for given :math:`\\omega^>` proposals.

            :param omega_mn: proposals; shape = (n_particles, batch_size)
            :return: log likelihood; shape = (n_particles, batch_size)
            """
            log_omega_mn = omega_mn.log()
            log_rate_combined_mnr = logaddexp(
                log_rate_e_lo_nr,
                log_rate_e_hi_nr + log_omega_mn.unsqueeze(-1))
            fingerprint_log_rate_prod_mn = (fingerprint_tensor_nr * log_rate_combined_mnr).sum(-1)
            return (fingerprint_log_rate_prod_mn
                    - mu_e_lo_n * prob_obs_lo_n
                    - omega_mn * mu_e_hi_n * prob_obs_hi_n
                    - fingerprint_log_norm_factor_n)

        def log_e_hi_conditional_moments_function(omega_mn: torch.Tensor) -> torch.Tensor:
            """Calculates the mean and the variance of gene expression over the :math:`\\omega^>` proposals
            conditional on not being zero-inflated.

            :param omega_mn: proposals; shape = (n_particles, batch_size)
            :return: a tensor of shape shape = (2, n_particles, batch_size); the leftmost dimension
                correspond to the first and second moments of gene expresion, in order.
            """
            log_omega_mn = omega_mn.log()
            log_rate_e_hi_mnr = log_rate_e_hi_nr + log_omega_mn.unsqueeze(-1)
            log_rate_combined_mnr = logaddexp(log_rate_e_lo_nr, log_rate_e_hi_mnr)

            log_p_binom_obs_hi_mnr = log_rate_e_hi_mnr - log_rate_combined_mnr
            log_p_binom_obs_lo_mnr = log_rate_e_lo_nr - log_rate_combined_mnr

            log_e_hi_obs_mom_1_mn = torch.logsumexp(
                log_fingerprint_tensor_nr + log_p_binom_obs_hi_mnr, -1)
            log_e_hi_obs_var_mn = torch.logsumexp(
                log_fingerprint_tensor_nr + log_p_binom_obs_hi_mnr + log_p_binom_obs_lo_mnr, -1)

            log_e_hi_unobs_mom_1_mn = log_mu_e_hi_n + log_omega_mn + log_prob_unobs_hi_n
            log_e_hi_unobs_var_mn = log_e_hi_unobs_mom_1_mn

            if run_mode == "full":
                log_e_hi_conditional_mom_1_mn = logaddexp(log_e_hi_unobs_mom_1_mn, log_e_hi_obs_mom_1_mn)
                log_e_hi_conditional_mom_2_mn = logaddexp(
                    2 * log_e_hi_conditional_mom_1_mn,
                    logaddexp(log_e_hi_unobs_var_mn, log_e_hi_obs_var_mn))
            elif run_mode == "only_observed":
                log_e_hi_conditional_mom_1_mn = log_e_hi_obs_mom_1_mn
                log_e_hi_conditional_mom_2_mn = logaddexp(
                    2 * log_e_hi_conditional_mom_1_mn,
                    log_e_hi_obs_var_mn)
            else:
                raise ValueError("Unknown run mode! valid choices are 'full' and 'only_observed'")

            return torch.stack((
                log_e_hi_conditional_mom_1_mn,
                log_e_hi_conditional_mom_2_mn), dim=0)

        # Gamma concentration and rates for prior and proposal distributions of :math:`\\omega`
        prior_concentration_n = alpha_e_hi_n
        prior_rate_n = alpha_e_hi_n
        proposal_concentration_n = alpha_e_hi_n + e_obs_n
        proposal_rate_n = alpha_e_hi_n + mu_e_hi_n * p_nonzero_e_hi_n
        omega_proposal_dist = dist.Gamma(proposal_concentration_n, proposal_rate_n)
        omega_prior_dist = dist.Gamma(prior_concentration_n, prior_rate_n)

        omega_importance_sampler_inputs = PosteriorImportanceSamplerInputs(
            proposal_distribution=omega_proposal_dist,
            prior_distribution=omega_prior_dist,
            log_likelihood_function=fingerprint_log_like_function,
            log_objective_function=log_e_hi_conditional_moments_function)

        return omega_importance_sampler_inputs, trained_model_context, minibatch_data

    def _estimate_log_gene_expression_with_fixed_n_particles(
            self,
            gene_index: int,
            cell_index_list: List[int],
            n_particles_omega: int,
            output_ess: bool,
            run_mode: str):
        """

        :param run_mode: see ``_generate_omega_importance_sampler_inputs``
        :param gene_index: index of the gene in the datastore
        :param cell_index_list: list of cell indices to analyse
        :param n_particles_omega: number of random proposals used for importance sampling
        :return: estimated first and second moments of gene expression; if output_ess == True, also the ESS of
            the two estimators, in order.
        """

        # generate inputs for importance sampling with no zero inflation and the posterior model context
        omega_importance_sampler_inputs, trained_model_context, _ = self._generate_omega_importance_sampler_inputs(
            gene_index=gene_index,
            cell_index_list=cell_index_list,
            n_particles_cell=1,
            run_mode=run_mode)

        # perform importance sampling assuming no zero inflation
        batch_size = len(cell_index_list)
        sampler = PosteriorImportanceSampler(omega_importance_sampler_inputs)
        sampler.run(
            n_particles=n_particles_omega,
            n_outputs=2,
            batch_shape=torch.Size([batch_size]))

        log_numerator_kn = sampler.log_numerator
        log_denominator_n = sampler.log_denominator
        log_mom_1_expression_numerator_n = log_numerator_kn[0, :]
        log_mom_2_expression_numerator_n = log_numerator_kn[1, :]

        logit_p_zero_e_hi_n: torch.Tensor = trained_model_context["logit_p_zero_e_hi_n"]
        log_p_zero_e_hi_n: torch.Tensor = torch.nn.functional.logsigmoid(logit_p_zero_e_hi_n)
        log_p_nonzero_e_hi_n = get_log_prob_compl(log_p_zero_e_hi_n)
        omega_zero_mn = torch.zeros((1, batch_size), device=self.device, dtype=self.dtype)
        log_fingerprint_likelihood_zero_n = omega_importance_sampler_inputs.log_likelihood_function(
            omega_zero_mn).squeeze(0)
        log_n_particles = np.log(n_particles_omega)
        log_zero_inflated_denominator_n = logaddexp(
            log_p_zero_e_hi_n + log_fingerprint_likelihood_zero_n,
            log_p_nonzero_e_hi_n + log_denominator_n - log_n_particles)

        log_zero_inflated_mom_1_expression_n = (
                log_p_nonzero_e_hi_n
                + log_mom_1_expression_numerator_n
                - log_n_particles
                - log_zero_inflated_denominator_n)
        log_zero_inflated_mom_2_expression_n = (
                log_p_nonzero_e_hi_n
                + log_mom_2_expression_numerator_n
                - log_n_particles
                - log_zero_inflated_denominator_n)

        if output_ess:
            combined_expression_ess_kn = sampler.ess
            log_mom_1_expression_ess_n = combined_expression_ess_kn[0, :]
            log_mom_2_expression_ess_n = combined_expression_ess_kn[0, :]

            return (log_zero_inflated_mom_1_expression_n,
                    log_zero_inflated_mom_2_expression_n,
                    log_mom_1_expression_ess_n,
                    log_mom_2_expression_ess_n)

        else:

            return (log_zero_inflated_mom_1_expression_n,
                    log_zero_inflated_mom_2_expression_n)

    def get_gene_expression_posterior_moments_summary(
            self,
            gene_index: int,
            n_particles_omega: int,
            n_particles_cell: int,
            cell_shard_size: int,
            only_expressing_cells: bool,
            run_mode: str = 'full',
            mode_estimation_strategy: str = 'lower_bound') -> Dict[str, np.ndarray]:
        """Calculate posterior gene expression summary/

        :param gene_index: index of the gene in the datastore
        :param n_particles_omega: number of random proposals used for importance sampling of :math:`\\omega^>`
        :param n_particles_cell: number of posterior samples from the guide
        :param cell_shard_size: how many cells to include in every batch
        :param only_expressing_cells: only analyse expressing cells
        :param run_mode: see ``_generate_omega_importance_sampler_inputs``
        :param mode_estimation_strategy: choices include 'upper_bound' and 'lower_bound'
        :return: a dictionary of summary statistics
        """

        assert mode_estimation_strategy in {'lower_bound', 'upper_bound'}

        def __fix_scalar(a: np.ndarray):
            if a.ndim == 0:
                a = a[None]
            return a

        if mode_estimation_strategy == 'lower_bound':
            def __get_approximate_mode(mean: torch.Tensor, std: torch.Tensor):
                return torch.clamp(torch.ceil(mean - np.sqrt(3) * std), 0)
        elif mode_estimation_strategy == 'upper_bound':
            def __get_approximate_mode(mean: torch.Tensor, std: torch.Tensor):
                return torch.clamp(torch.floor(mean + np.sqrt(3) * std), 0)
        else:
            raise ValueError("Invalid mode_estimation_strategy; valid options are `lower_bound` and `upper_bound`")

        e_hi_mean_shard_list = []
        e_hi_std_shard_list = []
        e_hi_map_shard_list = []
        included_cell_indices = []

        for cell_index_list in self._sharded_cell_index_generator(gene_index, cell_shard_size, only_expressing_cells):
            included_cell_indices += cell_index_list

            log_e_hi_mom_1_n_list = []
            log_e_hi_mom_2_n_list = []
            for _ in range(n_particles_cell):
                log_e_hi_mom_1_n, log_e_hi_mom_2_n = self._estimate_log_gene_expression_with_fixed_n_particles(
                    gene_index=gene_index,
                    cell_index_list=cell_index_list,
                    n_particles_omega=n_particles_omega,
                    output_ess=False,
                    run_mode=run_mode)
                log_e_hi_mom_1_n_list.append(log_e_hi_mom_1_n)
                log_e_hi_mom_2_n_list.append(log_e_hi_mom_2_n)
            log_e_hi_mom_1_n = torch.logsumexp(torch.stack(log_e_hi_mom_1_n_list), 0) - np.log(n_particles_cell)
            log_e_hi_mom_2_n = torch.logsumexp(torch.stack(log_e_hi_mom_2_n_list), 0) - np.log(n_particles_cell)

            mean_e_hi_n = log_e_hi_mom_1_n.exp()
            std_e_hi_n = torch.clamp(log_e_hi_mom_2_n.exp() - mean_e_hi_n.pow(2), 0).sqrt()
            map_e_hi_n = __get_approximate_mode(mean_e_hi_n, std_e_hi_n)

            e_hi_mean_shard_list.append(__fix_scalar(mean_e_hi_n.cpu().numpy()))
            e_hi_std_shard_list.append(__fix_scalar(std_e_hi_n.cpu().numpy()))
            e_hi_map_shard_list.append(__fix_scalar(map_e_hi_n.int().cpu().numpy()))

        e_hi_mean = np.zeros((self.dtm_model.sc_fingerprint_dtm.n_cells,), dtype=np.float)
        e_hi_std = np.zeros((self.dtm_model.sc_fingerprint_dtm.n_cells,), dtype=np.float)
        e_hi_map = np.zeros((self.dtm_model.sc_fingerprint_dtm.n_cells,), dtype=np.int)

        e_hi_mean[included_cell_indices] = np.concatenate(e_hi_mean_shard_list)
        e_hi_std[included_cell_indices] = np.concatenate(e_hi_std_shard_list)
        e_hi_map[included_cell_indices] = np.concatenate(e_hi_map_shard_list)

        return {'e_hi_map': e_hi_map,
                'e_hi_std': e_hi_std,
                'e_hi_mean': e_hi_mean}

    # TODO unit test
    @staticmethod
    def generate_zero_inflated_bootstrap_posterior_samples(
            proposal_proper_distribution: torch.distributions.Distribution,
            prior_proper_distribution: torch.distributions.Distribution,
            log_likelihood_function: Callable[[torch.Tensor], torch.Tensor],
            logit_prob_zero_prior_n: torch.Tensor,
            batch_size: int,
            n_proposals: int,
            n_particles: int,
            device: torch.device,
            dtype: torch.dtype) -> torch.Tensor:
        """Draws posterior samples from a zero-inflated prior distribution with a given likelihood function
        via bootstrap re-sampling.

        .. note:: we refer to the non-zero-inflated component of distributions as the *proper* part.
        """
        # draw proposals
        proposal_points_mn = proposal_proper_distribution.sample((n_proposals,))

        # calculate the log norm factor of the proper part of :math:`\\omega` posterior
        prior_proper_log_prob_mn = prior_proper_distribution.log_prob(proposal_points_mn)
        proposal_proper_log_prob_mn = proposal_proper_distribution.log_prob(proposal_points_mn)
        fingerprint_log_likelihood_mn = log_likelihood_function(proposal_points_mn)
        log_proper_posterior_norm_n = torch.logsumexp(
            prior_proper_log_prob_mn
            + fingerprint_log_likelihood_mn
            - proposal_proper_log_prob_mn, 0) - np.log(n_proposals)

        # prior zero-inflation log prob
        log_prob_zero_prior_n: torch.Tensor = torch.nn.functional.logsigmoid(logit_prob_zero_prior_n)
        log_prob_nonzero_prior_n = get_log_prob_compl(log_prob_zero_prior_n)

        # log likelihood at zero
        log_likelihood_zero_n = log_likelihood_function(
            torch.zeros((1, batch_size), device=device, dtype=dtype)).squeeze(0)

        # log normalization factor of the posterior, including the contribution of zero-inflation
        log_full_posterior_norm_n = logaddexp(
            log_prob_zero_prior_n + log_likelihood_zero_n,
            log_prob_nonzero_prior_n + log_proper_posterior_norm_n)

        # zero-inflation posterior log probability
        log_prob_zero_posterior_n = (
                log_prob_zero_prior_n
                + log_likelihood_zero_n
                - log_full_posterior_norm_n)
        logit_prob_nonzero_posterior_n = (
                get_log_prob_compl(log_prob_zero_posterior_n)
                - log_prob_zero_posterior_n)

        # draw proper posterior samples via bootstrap re-sampling
        log_posterior_proper_mn = (
            prior_proper_log_prob_mn
            + fingerprint_log_likelihood_mn
            - proposal_proper_log_prob_mn
            - log_proper_posterior_norm_n)
        posterior_proper_bootstrap_sample_indices_mn = torch.distributions.Categorical(
            logits=log_posterior_proper_mn.permute((1, 0))).sample((n_particles,))
        posterior_proper_bootstrap_samples_mn = torch.gather(
            proposal_points_mn,
            dim=0,
            index=posterior_proper_bootstrap_sample_indices_mn)

        # draw zero-inflation Bernoulli mask
        posterior_zero_inflation_mask_mn = torch.distributions.Bernoulli(
            logits=logit_prob_nonzero_posterior_n).sample((n_particles,)).float()

        return posterior_zero_inflation_mask_mn * posterior_proper_bootstrap_samples_mn

    def _generate_gene_expression_posterior_samples(
            self,
            gene_index: int,
            cell_index_list: List[int],
            n_proposals_omega: int,
            n_particles_omega: int,
            n_particles_cell: int,
            n_particles_expression: int,
            run_mode: str) -> torch.Tensor:
        """

        .. note:: "proper" means non-zero-inflated part throughput.

        :param gene_index:
        :param cell_index_list:
        :param n_proposals_omega:
        :param n_particles_omega:
        :param n_particles_cell:
        :param n_particles_expression:
        :param run_mode:
        :return:
        """

        # draw posterior samples from all non-marginalized latent variables
        (omega_importance_sampler_inputs,
         trained_model_context,
         minibatch_data) = self._generate_omega_importance_sampler_inputs(
            gene_index=gene_index,
            cell_index_list=cell_index_list,
            n_particles_cell=n_particles_cell,
            run_mode=run_mode)
        n_cells = len(cell_index_list)
        batch_size = n_particles_cell * n_cells

        omega_posterior_samples_mn = self.generate_zero_inflated_bootstrap_posterior_samples(
            proposal_proper_distribution=omega_importance_sampler_inputs.proposal_distribution,
            prior_proper_distribution=omega_importance_sampler_inputs.prior_distribution,
            log_likelihood_function=omega_importance_sampler_inputs.log_likelihood_function,
            logit_prob_zero_prior_n=trained_model_context["logit_p_zero_e_hi_n"],
            batch_size=batch_size,
            n_proposals=n_proposals_omega,
            n_particles=n_particles_omega,
            device=self.device,
            dtype=self.dtype)

        # fetch intermediate tensors
        log_prob_fsd_hi_full_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_full_nr"]
        log_prob_fsd_lo_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_lo_obs_nr"]
        log_prob_fsd_hi_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_obs_nr"]
        mu_e_lo_n: torch.Tensor = trained_model_context["mu_e_lo_n"]
        mu_e_hi_n: torch.Tensor = trained_model_context["mu_e_hi_n"]
        fingerprint_tensor_nr: torch.Tensor = minibatch_data["fingerprint_tensor"]

        # the probability of being a real molecule for every omega particle and family size
        log_mu_e_lo_n = mu_e_lo_n.log()
        log_mu_e_hi_n = mu_e_hi_n.log()
        log_rate_e_lo_nr = log_mu_e_lo_n.unsqueeze(-1) + log_prob_fsd_lo_obs_nr
        log_rate_e_hi_nr = log_mu_e_hi_n.unsqueeze(-1) + log_prob_fsd_hi_obs_nr
        log_omega_mn = omega_posterior_samples_mn.log()
        log_rate_e_hi_mnr = log_rate_e_hi_nr + log_omega_mn.unsqueeze(-1)
        log_rate_combined_mnr = logaddexp(log_rate_e_lo_nr, log_rate_e_hi_mnr)
        log_p_binom_obs_hi_mnr = log_rate_e_hi_mnr - log_rate_combined_mnr
        logit_p_binom_obs_hi_mnr = log_p_binom_obs_hi_mnr - get_log_prob_compl(log_p_binom_obs_hi_mnr)

        # draw posterior gene expression samples
        e_hi_obs_samples_smnr = get_binomial_samples_sparse_counts(
            total_counts=fingerprint_tensor_nr,
            logits=logit_p_binom_obs_hi_mnr,
            sample_shape=torch.Size((n_particles_expression,)))

        if run_mode == "only_observed":
            e_hi_posterior_samples_smn = e_hi_obs_samples_smnr.sum(-1)
        elif run_mode == "full":
            # the poisson rate for unobserved real molecules
            p_unobs_hi_n = log_prob_fsd_hi_full_nr[:, 0].exp()
            e_hi_unobs_dist = torch.distributions.Poisson(
                rate=p_unobs_hi_n * omega_posterior_samples_mn * mu_e_hi_n)
            e_hi_unobs_samples_smn = e_hi_unobs_dist.sample((n_particles_expression,))
            e_hi_posterior_samples_smn = e_hi_obs_samples_smnr.sum(-1) + e_hi_unobs_samples_smn
        else:
            raise ValueError("Unknown run mode! valid choices are 'full' and 'only_observed'")

        return e_hi_posterior_samples_smn \
            .int() \
            .permute(2, 1, 0) \
            .contiguous() \
            .view(n_cells, -1) \
            .permute(1, 0)

    def get_gene_expression_posterior_sampling_summary(
            self,
            gene_index: int,
            n_proposals_omega: int,
            n_particles_omega: int,
            n_particles_cell: int,
            n_particles_expression: int,
            cell_shard_size: int,
            run_mode: str,
            only_expressing_cells: bool) -> Dict[str, np.ndarray]:
        """

        :param gene_index:
        :param n_proposals_omega:
        :param n_particles_omega:
        :param n_particles_cell:
        :param n_particles_expression:
        :param cell_shard_size:
        :param run_mode:
        :param only_expressing_cells:
        :return:
        """

        def __fix_scalar(a: np.ndarray):
            if a.ndim == 0:
                a = a[None]
            return a

        e_hi_mean_shard_list = []
        e_hi_std_shard_list = []
        e_hi_map_shard_list = []
        included_cell_indices = []

        for cell_index_list in self._sharded_cell_index_generator(gene_index, cell_shard_size, only_expressing_cells):
            included_cell_indices += cell_index_list

            e_hi_posterior_samples_sn = self._generate_gene_expression_posterior_samples(
                gene_index=gene_index,
                cell_index_list=cell_index_list,
                n_proposals_omega=n_proposals_omega,
                n_particles_omega=n_particles_omega,
                n_particles_cell=n_particles_cell,
                n_particles_expression=n_particles_expression,
                run_mode=run_mode)

            e_hi_mean_shard_list.append(__fix_scalar(
                torch.mean(e_hi_posterior_samples_sn.float(), dim=0).cpu().numpy()))
            e_hi_std_shard_list.append(__fix_scalar(
                torch.std(e_hi_posterior_samples_sn.float(), dim=0).cpu().numpy()))
            e_hi_map_shard_list.append(__fix_scalar(
                int_ndarray_mode(e_hi_posterior_samples_sn.int().cpu().numpy(), axis=0)))

        e_hi_mean = np.zeros((self.dtm_model.sc_fingerprint_dtm.n_cells,), dtype=np.float)
        e_hi_std = np.zeros((self.dtm_model.sc_fingerprint_dtm.n_cells,), dtype=np.float)
        e_hi_map = np.zeros((self.dtm_model.sc_fingerprint_dtm.n_cells,), dtype=np.int)

        e_hi_mean[included_cell_indices] = np.concatenate(e_hi_mean_shard_list)
        e_hi_std[included_cell_indices] = np.concatenate(e_hi_std_shard_list)
        e_hi_map[included_cell_indices] = np.concatenate(e_hi_map_shard_list)

        return {'e_hi_map': e_hi_map,
                'e_hi_std': e_hi_std,
                'e_hi_mean': e_hi_mean}
