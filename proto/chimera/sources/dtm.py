import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator, Optional

from boltons.cacheutils import cachedproperty

import pyro
from pyro import poutine
import pyro.distributions as dist

import torch
from torch.distributions import constraints

from pyro_extras import CustomLogProbTerm, logaddexp, get_log_prob_compl, \
    get_binomial_samples_sparse_counts
from fingerprint import SingleCellFingerprintDTM
from fsd import FSDModel
from expr import GeneExpressionModel
from chimera import ChimeraRateModel
from importance_sampling import PosteriorImportanceSamplerInputs, PosteriorImportanceSampler
from stats import int_ndarray_mode
from utils import get_cell_averaged_from_collapsed_samples, get_detached_on_non_inducing_genes

import logging


class DropletTimeMachineModel(torch.nn.Module):
    def __init__(self,
                 init_params_dict: Dict[str, Union[int, float, bool]],
                 model_constraint_params_dict: Dict[str, Dict[str, Union[int, float]]],
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 gene_expression_model: GeneExpressionModel,
                 chimera_rate_model: ChimeraRateModel,
                 fsd_model: FSDModel,
                 monitor_constraint_pressure: bool = False,
                 constraint_pressure_forgetting_factor: float = 0.995,
                 device=torch.device('cuda'),
                 dtype=torch.float):
        super(DropletTimeMachineModel, self).__init__()

        self.model_constraint_params_dict = model_constraint_params_dict
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.gene_expression_model = gene_expression_model
        self.chimera_rate_model = chimera_rate_model
        self.fsd_model = fsd_model

        self.monitor_constraint_pressure = monitor_constraint_pressure
        self.constraint_pressure_forgetting_factor = constraint_pressure_forgetting_factor

        self.device = device
        self.dtype = dtype

        # hyperparameters
        self.enable_fsd_w_dirichlet_reg: bool = init_params_dict['fsd.enable_fsd_w_dirichlet_reg']
        self.w_lo_dirichlet_reg_strength: float = init_params_dict['fsd.w_lo_dirichlet_reg_strength']
        self.w_hi_dirichlet_reg_strength: float = init_params_dict['fsd.w_hi_dirichlet_reg_strength']
        self.w_lo_dirichlet_concentration: float = init_params_dict['fsd.w_lo_dirichlet_concentration']
        self.w_hi_dirichlet_concentration: float = init_params_dict['fsd.w_hi_dirichlet_concentration']
        self.fsd_xi_posterior_min_scale: float = init_params_dict['fsd.xi_posterior_min_scale']
        self.n_particles_fingerprint_log_like: int = init_params_dict['model.n_particles_fingerprint_log_like']

        self.enable_chimera_rate_auto_regularization: bool = \
            init_params_dict['chimera.enable_rate_auto_regularization']
        self.chimera_rate_auto_regularization_strength: float = \
            init_params_dict['chimera.rate_auto_regularization_strength']

        # empirical normalization factors
        self.mean_empirical_fsd_mu_hi: float = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi).item()

        # logging
        self._logger = logging.getLogger()

        # todo refactor and allow arbitrary list of genes
        # a binary mask for highly variable genes
        self.inducing_binary_mask_tensor_g = torch.tensor(
            sc_fingerprint_dtm.hvg_binary_mask, device=device, dtype=torch.bool)

        # constraint pressure
        if self.monitor_constraint_pressure:
            self.constraint_pressure_dict = dict()
            for var_name, var_constraint_params in model_constraint_params_dict.items():
                self.constraint_pressure_dict[var_name] = dict()
                if 'lower_bound_value' in var_constraint_params:
                    self.constraint_pressure_dict[var_name]['lower_bound_pressure'] = 0.0
                if 'upper_bound_value' in var_constraint_params:
                    self.constraint_pressure_dict[var_name]['upper_bound_pressure'] = 0.0
                if 'pin_value' in var_constraint_params:
                    self.constraint_pressure_dict[var_name]['pinning_pressure'] = 0.0

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
        assert 'fingerprint_tensor' in data
        assert 'gene_sampling_site_scale_factor_tensor' in data
        assert 'cell_sampling_site_scale_factor_tensor' in data
        assert 'empirical_fsd_mu_hi_tensor' in data
        assert 'arithmetic_mean_obs_expr_per_gene_tensor' in data
        assert 'gene_index_tensor' in data
        assert 'cell_index_tensor' in data
        assert 'cell_features_tensor' in data
        assert 'empirical_droplet_efficiency_tensor' in data

        # input tensors
        fingerprint_tensor_nr = data['fingerprint_tensor']
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        cell_sampling_site_scale_factor_tensor_n = data['cell_sampling_site_scale_factor_tensor']
        empirical_mu_fsd_tensor_n = data['empirical_fsd_mu_hi_tensor']  # TODO fix tensor name (drop hi)
        arithmetic_mean_obs_expr_per_gene_tensor_n = data['arithmetic_mean_obs_expr_per_gene_tensor']
        gene_index_tensor_n = data['gene_index_tensor']
        cell_index_tensor_n = data['cell_index_tensor']
        cell_features_tensor_nf = data['cell_features_tensor']
        eta_n = data['empirical_droplet_efficiency_tensor']

        # sizes
        mb_size = fingerprint_tensor_nr.shape[0]
        batch_shape = torch.Size([mb_size])
        max_family_size = fingerprint_tensor_nr.shape[1]

        # register the external modules
        pyro.module("fsd_model", self.fsd_model, update_module_params=True)
        pyro.module("gene_expression_model", self.gene_expression_model, update_module_params=True)
        pyro.module("chimera_rate_model", self.chimera_rate_model, update_module_params=True)

        # useful auxiliary quantities
        family_size_vector_obs_r = torch.arange(
            1, max_family_size + 1, device=self.device, dtype=self.dtype)
        family_size_vector_full_r = torch.arange(
            0, max_family_size + 1, device=self.device, dtype=self.dtype)
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)

        # inducing binary mask
        inducing_binary_mask_tensor_n = self.inducing_binary_mask_tensor_g[gene_index_tensor_n].float()
        non_inducing_binary_mask_tensor_n = (~self.inducing_binary_mask_tensor_g[gene_index_tensor_n]).float()

        # sample unconstrained fsd params
        fsd_model_output_dict = self.fsd_model.model(data)

        # decode to constrained fsd params
        fsd_params_dict = self.fsd_model.decode(
            output_dict=fsd_model_output_dict,
            parents_dict={
                'inducing_binary_mask_tensor_n': inducing_binary_mask_tensor_n,
                'non_inducing_binary_mask_tensor_n': non_inducing_binary_mask_tensor_n
            })

        # get chimeric and real family size distributions
        fsd_lo_dist, fsd_hi_dist = self.fsd_model.get_fsd_components(fsd_params_dict)

        # get e_hi prior parameters (per cell)
        gene_expression_model_output_dict = self.gene_expression_model.model(data)

        # calculate NB parameters
        e_hi_nb_params_dict = self.gene_expression_model.decode_output_to_nb_params_dict(
            output_dict=gene_expression_model_output_dict,
            data=data)

        log_mu_e_hi_n = e_hi_nb_params_dict['log_mu_e_hi_n']
        log_phi_e_hi_n = e_hi_nb_params_dict['log_phi_e_hi_n']

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

        mu_fsd_lo_zero_truncated_n = mu_fsd_lo_n / p_obs_lo_n
        mu_fsd_hi_zero_truncated_n = mu_fsd_hi_n / p_obs_hi_n
        mu_fsd_lo_zero_truncated_to_mu_fsd_empirical_ratio_n =\
            mu_fsd_lo_zero_truncated_n / empirical_mu_fsd_tensor_n
        mu_fsd_hi_zero_truncated_to_mu_fsd_empirical_ratio_n =\
            mu_fsd_hi_zero_truncated_n / empirical_mu_fsd_tensor_n
        
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

        # sample chimera parameters
        chimera_rate_model_output_dict = self.chimera_rate_model.model(data)

        # extract chimera rate parameters
        chimera_rate_params_dict = self.chimera_rate_model.decode_output_to_chimera_rate(
            output_dict=chimera_rate_model_output_dict,
            data_dict=data,
            parents_dict={
                'mu_e_hi_n': mu_e_hi_n,
                'gene_index_tensor_n': gene_index_tensor_n,
                'cell_sampling_site_scale_factor_tensor_n': cell_sampling_site_scale_factor_tensor_n,
                'mu_fsd_hi_n': mu_fsd_hi_n,
                'eta_n': eta_n,
                'inducing_binary_mask_tensor_n': inducing_binary_mask_tensor_n,
                'non_inducing_binary_mask_tensor_n': non_inducing_binary_mask_tensor_n
            })
        mu_e_lo_n = chimera_rate_params_dict['mu_e_lo_n']

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
                phi_e_lo_n=None,
                mu_e_hi_n=mu_e_hi_n,
                phi_e_hi_n=phi_e_hi_n,
                n_particles=self.n_particles_fingerprint_log_like)

            # sample fsd sparsity regularization
            if self.enable_fsd_w_dirichlet_reg:
                self._sample_fsd_weight_sparsity_regularization(
                    w_lo_dirichlet_reg_strength=self.w_lo_dirichlet_reg_strength,
                    w_hi_dirichlet_reg_strength=self.w_hi_dirichlet_reg_strength,
                    w_lo_dirichlet_concentration=self.w_lo_dirichlet_concentration,
                    w_hi_dirichlet_concentration=self.w_hi_dirichlet_concentration,
                    n_fsd_lo_comps=self.fsd_model.n_fsd_lo_comps,
                    n_fsd_hi_comps=self.fsd_model.n_fsd_hi_comps,
                    w_fsd_lo_comps_nj=w_fsd_lo_comps_nj,
                    w_fsd_hi_comps_nj=w_fsd_hi_comps_nj,
                    gene_sampling_site_scale_factor_tensor_n=gene_sampling_site_scale_factor_tensor_n)

            # sample (soft) constraints
            total_obs_expr_per_gene_tensor_n = \
                self.sc_fingerprint_dtm.n_cells * arithmetic_mean_obs_expr_per_gene_tensor_n
            self._sample_gene_plate_soft_constraints(
                model_constraint_params_dict=self.model_constraint_params_dict,
                model_vars_dict=locals(),
                gene_sampling_site_scale_factor_tensor_n=gene_sampling_site_scale_factor_tensor_n,
                total_obs_expr_per_gene_tensor_n=total_obs_expr_per_gene_tensor_n,
                batch_shape=batch_shape)

            # observed chimeric fraction regularization
            if self.enable_chimera_rate_auto_regularization:
                self._sample_chimera_fraction_autoreg(
                    mu_e_lo_n=mu_e_lo_n,
                    mu_e_hi_n=mu_e_hi_n,
                    p_obs_lo_n=p_obs_lo_n,
                    p_obs_hi_n=p_obs_hi_n,
                    inducing_binary_mask_tensor_n=inducing_binary_mask_tensor_n,
                    non_inducing_binary_mask_tensor_n=non_inducing_binary_mask_tensor_n,
                    gene_index_tensor_n=gene_index_tensor_n,
                    gene_sampling_site_scale_factor_tensor_n=gene_sampling_site_scale_factor_tensor_n,
                    cell_sampling_site_scale_factor_tensor_n=cell_sampling_site_scale_factor_tensor_n,
                    auto_regularization_strength=self.chimera_rate_auto_regularization_strength)

    def _sample_chimera_fraction_autoreg(self,
                                         mu_e_lo_n: torch.Tensor,
                                         mu_e_hi_n: torch.Tensor,
                                         p_obs_lo_n: torch.Tensor,
                                         p_obs_hi_n: torch.Tensor,
                                         inducing_binary_mask_tensor_n: torch.Tensor,
                                         non_inducing_binary_mask_tensor_n: torch.Tensor,
                                         gene_index_tensor_n: torch.Tensor,
                                         gene_sampling_site_scale_factor_tensor_n: torch.Tensor,
                                         cell_sampling_site_scale_factor_tensor_n: torch.Tensor,
                                         auto_regularization_strength: float):
        """Regularize chimeric fraction of non-inducing genes with that of inducing genes"""
        e_lo_obs_cell_averaged_n = get_cell_averaged_from_collapsed_samples(
            input_tensor_n=mu_e_lo_n * p_obs_lo_n,
            gene_index_tensor_n=gene_index_tensor_n,
            cell_sampling_site_scale_factor_tensor_n=cell_sampling_site_scale_factor_tensor_n,
            n_genes=self.sc_fingerprint_dtm.n_genes,
            dtype=self.dtype,
            device=self.device)

        e_hi_obs_cell_averaged_n = get_cell_averaged_from_collapsed_samples(
            input_tensor_n=mu_e_hi_n * p_obs_hi_n,
            gene_index_tensor_n=gene_index_tensor_n,
            cell_sampling_site_scale_factor_tensor_n=cell_sampling_site_scale_factor_tensor_n,
            n_genes=self.sc_fingerprint_dtm.n_genes,
            dtype=self.dtype,
            device=self.device)

        prior_chimera_fraction_n = e_lo_obs_cell_averaged_n / (
                e_lo_obs_cell_averaged_n + e_hi_obs_cell_averaged_n)

        # todo magic number
        prior_chimera_fraction_alpha = pyro.param(
            "prior_chimera_fraction_alpha",
            torch.tensor(1.1, device=self.device, dtype=self.dtype),
            constraint=constraints.greater_than_eq(1.))

        # todo magic number
        prior_chimera_fraction_beta = pyro.param(
            "prior_chimera_fraction_beta",
            torch.tensor(1.1, device=self.device, dtype=self.dtype),
            constraint=constraints.greater_than_eq(1.))

        prior_chimera_fraction_alpha_n = get_detached_on_non_inducing_genes(
            input_scalar=prior_chimera_fraction_alpha,
            inducing_binary_mask_tensor_n=inducing_binary_mask_tensor_n,
            non_inducing_binary_mask_tensor_n=non_inducing_binary_mask_tensor_n)

        prior_chimera_fraction_beta_n = get_detached_on_non_inducing_genes(
            input_scalar=prior_chimera_fraction_beta,
            inducing_binary_mask_tensor_n=inducing_binary_mask_tensor_n,
            non_inducing_binary_mask_tensor_n=non_inducing_binary_mask_tensor_n)

        with poutine.scale(scale=auto_regularization_strength * gene_sampling_site_scale_factor_tensor_n):
            pyro.sample(
                "prior_chimera_fraction_autoreg",
                dist.Beta(prior_chimera_fraction_alpha_n, prior_chimera_fraction_beta_n),
                obs=prior_chimera_fraction_n)

    def guide(self,
              data: Dict[str, torch.Tensor],
              posterior_sampling_mode: bool = False):

        # register the external modules
        pyro.module("fsd_model", self.fsd_model, update_module_params=True)
        pyro.module("gene_expression_model", self.gene_expression_model, update_module_params=True)
        pyro.module("chimera_rate_model", self.chimera_rate_model, update_module_params=True)

        # gene expression posterior parameters
        gene_expr_guide_output_dict = self.gene_expression_model.guide(data)

        # fsd posterior parameters
        fsd_guide_output_dict = self.fsd_model.guide(data)

        # sample chimera parameters
        chimera_rate_guide_output_dict = self.chimera_rate_model.guide(data)

        return {
            'gene_expr_guide_output_dict': gene_expr_guide_output_dict,
            'fsd_guide_output_dict': fsd_guide_output_dict,
            'chimera_rate_guide_output_dict': chimera_rate_guide_output_dict}

    @staticmethod
    def _sample_fingerprint(batch_shape: torch.Size,
                            cell_sampling_site_scale_factor_tensor_n: torch.Tensor,
                            fingerprint_tensor_nr: torch.Tensor,
                            log_prob_fsd_lo_obs_nr: torch.Tensor,
                            log_prob_fsd_hi_obs_nr: torch.Tensor,
                            mu_e_lo_n: torch.Tensor,
                            phi_e_lo_n: Optional[torch.Tensor],
                            mu_e_hi_n: torch.Tensor,
                            phi_e_hi_n: torch.Tensor,
                            n_particles: int):

        # calculate the fingerprint log likelihood
        fingerprint_log_likelihood_n = DropletTimeMachineModel._get_fingerprint_log_likelihood_monte_carlo(
            fingerprint_tensor_nr=fingerprint_tensor_nr,
            log_prob_fsd_lo_obs_nr=log_prob_fsd_lo_obs_nr,
            log_prob_fsd_hi_obs_nr=log_prob_fsd_hi_obs_nr,
            mu_e_lo_n=mu_e_lo_n,
            phi_e_lo_n=phi_e_lo_n,
            mu_e_hi_n=mu_e_hi_n,
            phi_e_hi_n=phi_e_hi_n,
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
    def _get_fingerprint_log_likelihood_monte_carlo(fingerprint_tensor_nr: torch.Tensor,
                                                    log_prob_fsd_lo_obs_nr: torch.Tensor,
                                                    log_prob_fsd_hi_obs_nr: torch.Tensor,
                                                    mu_e_lo_n: torch.Tensor,
                                                    phi_e_lo_n: Optional[torch.Tensor],
                                                    mu_e_hi_n: torch.Tensor,
                                                    phi_e_hi_n: torch.Tensor,
                                                    n_particles: int) -> torch.Tensor:
        """Calculates the approximate fingerprint log likelihood by marginalizing Gamma prior distribution
        of real gene expression rate via Monte-Carlo sampling.

        .. note:: Importantly, the samples drawn of the ZIG distribution must be differentiable
            (e.g. re-parametrized) w.r.t. to its parameters.

        .. note:: For the purposes of model fitting, the overall data-dependent normalization factor
            of the log likelihood is immaterial and we drop it.

        :param n_particles: number of MC samples to draw for marginalizing the ZIG prior for :math:`e^>`.
        :return: fingerprint log likelihood
        """

        assert phi_e_lo_n is None  # (debug)

        # pre-compute useful tensors
        p_lo_obs_nr = log_prob_fsd_lo_obs_nr.exp()
        total_obs_rate_lo_n = mu_e_lo_n * p_lo_obs_nr.sum(-1)
        log_rate_e_lo_nr = mu_e_lo_n.log().unsqueeze(-1) + log_prob_fsd_lo_obs_nr

        p_hi_obs_nr = log_prob_fsd_hi_obs_nr.exp()
        total_obs_rate_hi_n = mu_e_hi_n * p_hi_obs_nr.sum(-1)
        log_rate_e_hi_nr = mu_e_hi_n.log().unsqueeze(-1) + log_prob_fsd_hi_obs_nr

        fingerprint_log_norm_factor_n = (fingerprint_tensor_nr + 1).lgamma().sum(-1)

        # step 1. draw re-parametrized Gamma particles
        alpha_e_hi_n = phi_e_hi_n.reciprocal()
        omega_hi_mn = dist.Gamma(concentration=alpha_e_hi_n, rate=alpha_e_hi_n).rsample((n_particles,))

        # alpha_e_lo_n = phi_e_lo_n.reciprocal()
        # omega_lo_mn = dist.Gamma(concentration=alpha_e_lo_n, rate=alpha_e_lo_n).rsample((n_particles,))

        # step 2. calculate the conditional log likelihood for each of the Gamma particles
        log_rate_combined_mnr = logaddexp(
            log_rate_e_lo_nr,  # + omega_lo_mn.log().unsqueeze(-1),
            log_rate_e_hi_nr + omega_hi_mn.log().unsqueeze(-1))
        log_poisson_e_hi_mn = (
            (fingerprint_tensor_nr * log_rate_combined_mnr).sum(-1)
            - total_obs_rate_lo_n  # * omega_lo_mn
            - total_obs_rate_hi_n * omega_hi_mn
            - fingerprint_log_norm_factor_n)  # data-dependent norm factor can be dropped

        # step 3. average over the Gamma particles
        log_like_n = log_poisson_e_hi_mn.logsumexp(0) - np.log(n_particles)

        return log_like_n

    def _sample_gene_plate_soft_constraints(
            self,
            model_constraint_params_dict: Dict[str, Dict[str, float]],
            model_vars_dict: Dict[str, torch.Tensor],
            gene_sampling_site_scale_factor_tensor_n: torch.Tensor,
            total_obs_expr_per_gene_tensor_n: torch.Tensor,
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
                    constraint_log_prob *= total_obs_expr_per_gene_tensor_n
                    pyro.sample(
                        var_name + "_lower_bound_constraint",
                        CustomLogProbTerm(constraint_log_prob,
                                          batch_shape=batch_shape,
                                          event_shape=torch.Size([])),
                        obs=torch.zeros_like(constraint_log_prob))

                    if self.monitor_constraint_pressure:
                        self.constraint_pressure_dict[var_name]['lower_bound_pressure'] = (
                            self.constraint_pressure_forgetting_factor
                            * self.constraint_pressure_dict[var_name]['lower_bound_pressure']
                            + (1 - self.constraint_pressure_forgetting_factor)
                            * constraint_log_prob.abs().sum().item())

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
                    constraint_log_prob *= total_obs_expr_per_gene_tensor_n
                    pyro.sample(
                        var_name + "_upper_bound_constraint",
                        CustomLogProbTerm(constraint_log_prob,
                                          batch_shape=batch_shape,
                                          event_shape=torch.Size([])),
                        obs=torch.zeros_like(constraint_log_prob))

                    if self.monitor_constraint_pressure:
                        self.constraint_pressure_dict[var_name]['upper_bound_pressure'] = (
                                self.constraint_pressure_forgetting_factor
                                * self.constraint_pressure_dict[var_name]['upper_bound_pressure']
                                + (1 - self.constraint_pressure_forgetting_factor)
                                * constraint_log_prob.abs().sum().item())

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
                    constraint_log_prob *= total_obs_expr_per_gene_tensor_n
                    pyro.sample(
                        var_name + "_pin_value_constraint",
                        CustomLogProbTerm(constraint_log_prob,
                                          batch_shape=batch_shape,
                                          event_shape=torch.Size([])),
                        obs=torch.zeros_like(constraint_log_prob))

                    if self.monitor_constraint_pressure:
                        self.constraint_pressure_dict[var_name]['pinning_pressure'] = (
                                self.constraint_pressure_forgetting_factor
                                * self.constraint_pressure_dict[var_name]['pinning_pressure']
                                + (1 - self.constraint_pressure_forgetting_factor)
                                * constraint_log_prob.abs().sum().item())

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

    # TODO: rewrite using pyro.poutine.trace and avoid code repetition
    # @torch.no_grad()
    # def get_active_constraints_on_genes(self) -> Dict:
    #     # TODO grab variables from the model
    #     raise NotImplementedError
    #
    #     # model_vars_dict = ...
    #     active_constraints_dict = defaultdict(dict)
    #     for var_name, var_constraint_params in self.model_constraint_params_dict.items():
    #         var = model_vars_dict[var_name]
    #         if 'lower_bound_value' in var_constraint_params:
    #             value = var_constraint_params['lower_bound_value']
    #             width = var_constraint_params['lower_bound_width']
    #             if isinstance(value, str):
    #                 value = model_vars_dict[value]
    #             activity = torch.clamp(value + width - var, min=0.)
    #             for _ in range(len(var.shape) - 1):
    #                 activity = activity.sum(-1)
    #             nnz_activity = torch.nonzero(activity).cpu().numpy().flatten()
    #             if nnz_activity.size > 0:
    #                 active_constraints_dict[var_name]['lower_bound'] = set(nnz_activity.tolist())
    #
    #         if 'upper_bound_value' in var_constraint_params:
    #             value = var_constraint_params['upper_bound_value']
    #             width = var_constraint_params['upper_bound_width']
    #             if isinstance(value, str):
    #                 value = model_vars_dict[value]
    #             activity = torch.clamp(var - value + width, min=0.)
    #             for _ in range(len(var.shape) - 1):
    #                 activity = activity.sum(-1)
    #             nnz_activity = torch.nonzero(activity).cpu().numpy().flatten()
    #             if nnz_activity.size > 0:
    #                 active_constraints_dict[var_name]['upper_bound'] = set(nnz_activity.tolist())
    #
    #     return dict(active_constraints_dict)


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
        and variance of gene expression.

        .. note:: According to the model, the prior for :math:`\\mu^<`, the Poisson rate for chimeric
            molecules, is deterministic; as such, it is directly picked up from the trained model
            context ``trained_model_context["mu_e_lo_n"]`` and no marginalization is necessary.

            The prior :math:`\\mu^>`, the Poisson rate for real molecules, however, is Gamma and must be
            marginalized. This method generates importance sampling inputs for marginalizing :math:`\\mu^>`
            and calculating the first and second moments of of gene expression.. We parametrize :math:`\\mu^>`
            as follows:

            .. math::

                \\mu^> = \\mathbb{E}[\\mu^>] \\omega^>,

                \\omega^> \\sim \\mathrm{Gamma}(\\alpha^>, \\alpha^>),

                \\alpha^> = 1 / \\phi^>,

            where :math:`\\phi^>` is the prior overdispersion of expression and :math:`\\mathbb{E}[\\mu^>]`
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
        minibatch_data = self.dtm_model.sc_fingerprint_dtm.generate_single_gene_fingerprint_minibatch_data(
            gene_index=gene_index,
            cell_index_list=cell_index_list,
            n_particles_cell=n_particles_cell)
        trained_model_context = self._get_trained_model_context(minibatch_data)

        # localize required auxiliary quantities from the trained model context
        fingerprint_tensor_nr = minibatch_data["fingerprint_tensor"]
        mu_e_hi_n: torch.Tensor = trained_model_context["mu_e_hi_n"]
        phi_e_hi_n: torch.Tensor = trained_model_context["phi_e_hi_n"]
        mu_e_lo_n: torch.Tensor = trained_model_context["mu_e_lo_n"]
        log_prob_fsd_lo_full_nr: torch.Tensor = trained_model_context["log_prob_fsd_lo_full_nr"]
        log_prob_fsd_hi_full_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_full_nr"]
        log_prob_fsd_lo_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_lo_obs_nr"]
        log_prob_fsd_hi_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_obs_nr"]

        # calculate additional auxiliary quantities
        alpha_e_hi_n = phi_e_hi_n.reciprocal()

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
            """Calculates the mean and the variance of gene expression over the :math:`\\omega^>` proposals.

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
        proposal_rate_n = alpha_e_hi_n + mu_e_hi_n
        omega_proposal_dist = dist.Gamma(proposal_concentration_n, proposal_rate_n)
        omega_prior_dist = dist.Gamma(prior_concentration_n, prior_rate_n)

        omega_importance_sampler_inputs = PosteriorImportanceSamplerInputs(
            proposal_distribution=omega_proposal_dist,
            prior_distribution=omega_prior_dist,
            log_likelihood_function=fingerprint_log_like_function,
            log_objective_function=log_e_hi_conditional_moments_function)

        return omega_importance_sampler_inputs, trained_model_context, minibatch_data

    @torch.no_grad()
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

        # generate inputs for posterior importance sampling and the posterior model context
        omega_importance_sampler_inputs, trained_model_context, _ = self._generate_omega_importance_sampler_inputs(
            gene_index=gene_index,
            cell_index_list=cell_index_list,
            n_particles_cell=1,
            run_mode=run_mode)

        # perform importance sampling
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

        log_mom_1_expression_n = log_mom_1_expression_numerator_n - log_denominator_n
        log_mom_2_expression_n = log_mom_2_expression_numerator_n - log_denominator_n

        if output_ess:
            combined_expression_ess_kn = sampler.ess
            log_mom_1_expression_ess_n = combined_expression_ess_kn[0, :]
            log_mom_2_expression_ess_n = combined_expression_ess_kn[0, :]

            return (log_mom_1_expression_n,
                    log_mom_2_expression_n,
                    log_mom_1_expression_ess_n,
                    log_mom_2_expression_ess_n)

        else:

            return (log_mom_1_expression_n,
                    log_mom_2_expression_n)

    @torch.no_grad()
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
    # TODO move to importance sampling?
    @staticmethod
    def generate_bootstrap_posterior_samples(
            proposal_distribution: torch.distributions.Distribution,
            prior_distribution: torch.distributions.Distribution,
            log_likelihood_function: Callable[[torch.Tensor], torch.Tensor],
            n_proposals: int,
            n_particles: int) -> torch.Tensor:
        """Draws posterior samples from a given prior, proposal, and likelihood function via bootstrap re-sampling.
        """
        # draw proposals
        proposal_points_mn = proposal_distribution.sample((n_proposals,))

        # calculate the log norm factor of the :math:`\\omega` posterior
        prior_log_prob_mn = prior_distribution.log_prob(proposal_points_mn)
        proposal_log_prob_mn = proposal_distribution.log_prob(proposal_points_mn)
        fingerprint_log_likelihood_mn = log_likelihood_function(proposal_points_mn)
        log_posterior_norm_n = torch.logsumexp(
            prior_log_prob_mn
            + fingerprint_log_likelihood_mn
            - proposal_log_prob_mn, 0) - np.log(n_proposals)

        # draw proper posterior samples via bootstrap re-sampling
        log_posterior_mn = (
            prior_log_prob_mn
            + fingerprint_log_likelihood_mn
            - proposal_log_prob_mn
            - log_posterior_norm_n)
        posterior_bootstrap_sample_indices_mn = torch.distributions.Categorical(
            logits=log_posterior_mn.permute((1, 0))).sample((n_particles,))
        posterior_bootstrap_samples_mn = torch.gather(
            proposal_points_mn,
            dim=0,
            index=posterior_bootstrap_sample_indices_mn)

        return posterior_bootstrap_samples_mn

    def _generate_gene_expression_posterior_samples(
            self,
            gene_index: int,
            cell_index_list: List[int],
            n_proposals_omega: int,
            n_particles_omega: int,
            n_particles_cell: int,
            n_particles_expression: int,
            run_mode: str) -> torch.Tensor:
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

        omega_posterior_samples_mn = self.generate_bootstrap_posterior_samples(
            proposal_distribution=omega_importance_sampler_inputs.proposal_distribution,
            prior_distribution=omega_importance_sampler_inputs.prior_distribution,
            log_likelihood_function=omega_importance_sampler_inputs.log_likelihood_function,
            n_proposals=n_proposals_omega,
            n_particles=n_particles_omega)

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
