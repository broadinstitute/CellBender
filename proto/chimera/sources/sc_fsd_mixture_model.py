import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable

import pyro
from pyro import poutine
import pyro.distributions as dist

import torch
from torch.distributions import constraints
from torch.nn.parameter import Parameter

from pyro_extras import CustomLogProbTerm, ZeroInflatedNegativeBinomial, \
    MixtureDistribution, logit, logaddexp, get_log_prob_compl
from sc_fingerprint import SingleCellFingerprintDataStore
from sc_fsd_codec import FamilySizeDistributionCodec, SortByComponentWeights
from sampling import PosteriorImportanceSamplerInputs, PosteriorImportanceSampler

import logging
from collections import defaultdict


class SingleCellFamilySizeModel(torch.nn.Module):

    EPS = 1e-6
    
    def __init__(self,
                 init_params_dict: dict,
                 model_constraint_params_dict: dict,
                 sc_fingerprint_datastore: SingleCellFingerprintDataStore,
                 fsd_codec: FamilySizeDistributionCodec,
                 guide_type: str = 'map',
                 device=torch.device('cuda'),
                 dtype=torch.float):
        super(SingleCellFamilySizeModel, self).__init__()

        self.model_constraint_params_dict = model_constraint_params_dict
        self.sc_fingerprint_datastore = sc_fingerprint_datastore
        self.fsd_codec = fsd_codec
        
        self.n_total_cells = sc_fingerprint_datastore.n_cells
        self.n_total_genes = sc_fingerprint_datastore.n_genes

        self.guide_type = guide_type

        self.device = device
        self.dtype = dtype

        # hyperparameters
        self.fsd_gmm_num_components = init_params_dict['fsd.gmm_num_components']
        self.fsd_gmm_dirichlet_concentration = init_params_dict['fsd.gmm_dirichlet_concentration']
        self.fsd_gmm_init_xi_scale = init_params_dict['fsd.gmm_init_xi_scale']
        self.fsd_gmm_min_xi_scale = init_params_dict['fsd.gmm_min_xi_scale']
        self.fsd_gmm_init_components_perplexity = init_params_dict['fsd.gmm_init_components_perplexity']
        self.fsd_gmm_min_weight_per_component = init_params_dict['fsd.gmm_min_weight_per_component']
        self.enable_fsd_w_dirichlet_reg = init_params_dict['fsd.enable_fsd_w_dirichlet_reg']
        self.w_lo_dirichlet_reg_strength = init_params_dict['fsd.w_lo_dirichlet_reg_strength']
        self.w_hi_dirichlet_reg_strength = init_params_dict['fsd.w_hi_dirichlet_reg_strength']
        self.w_lo_dirichlet_concentration = init_params_dict['fsd.w_lo_dirichlet_concentration']
        self.w_hi_dirichlet_concentration = init_params_dict['fsd.w_hi_dirichlet_concentration']
        self.train_chimera_rate_params = init_params_dict['chimera.enable_hyperparameter_optimization']
        self.fsd_xi_posterior_min_scale = init_params_dict['fsd.xi_posterior_min_scale']
        self.fingerprint_log_likelihood_n_particles = init_params_dict['model.fingerprint_log_likelihood_n_particles']

        # empirical normalization factors
        self.median_total_reads_per_cell = np.median(sc_fingerprint_datastore.total_obs_reads_per_cell)
        self.median_fsd_mu_hi = np.median(sc_fingerprint_datastore.empirical_fsd_mu_hi)

        # initial parameters for e_lo
        self.init_alpha_c = init_params_dict['chimera.alpha_c']
        self.init_beta_c = init_params_dict['chimera.beta_c']

        # initial parameters for e_hi
        self.init_mu_e_hi_g = sc_fingerprint_datastore.empirical_mu_e_hi
        self.init_phi_e_hi_g = sc_fingerprint_datastore.empirical_phi_e_hi
        self.init_logit_p_zero_e_hi_g = logit(torch.tensor(sc_fingerprint_datastore.empirical_p_zero_e_hi)).numpy()

        # logging
        self._logger = logging.getLogger()
                
    def forward(self, _):
        raise NotImplementedError

    def model(self,
              data,
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
        gene_index_tensor_n = data['gene_index_tensor']
        cell_index_tensor_n = data['cell_index_tensor']
        total_obs_reads_per_cell_tensor_n = data['total_obs_reads_per_cell_tensor']

        # sizes
        mb_size = fingerprint_tensor_nr.shape[0]
        batch_shape = torch.Size([mb_size])
        max_family_size = fingerprint_tensor_nr.shape[1]

        # register the parameters of the family size distribution codec
        pyro.module("fsd_codec", self.fsd_codec)
        
        # GMM prior for family size distribution parameters
        fsd_xi_prior_locs_kq = pyro.param(
            "fsd_xi_prior_locs_kq",
            self.fsd_codec.init_fsd_xi_loc_prior +
            self.fsd_gmm_init_components_perplexity * torch.randn(
                (self.fsd_gmm_num_components, self.fsd_codec.total_fsd_params),
                dtype=self.dtype, device=self.device))

        fsd_xi_prior_scales_kq = pyro.param(
            "fsd_xi_prior_scales_kq",
            self.fsd_gmm_init_xi_scale * torch.ones(
                (self.fsd_gmm_num_components, self.fsd_codec.total_fsd_params),
                dtype=self.dtype, device=self.device),
            constraint=constraints.greater_than(self.fsd_gmm_min_xi_scale))
        
        # chimera parameters
        alpha_c = pyro.param(
            "alpha_c",
            torch.tensor(self.init_alpha_c, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        beta_c = pyro.param(
            "beta_c",
            torch.tensor(self.init_beta_c, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)

        # gene expression parameters
        mu_e_hi_g = pyro.param(
            "mu_e_hi_g",
            torch.tensor(self.init_mu_e_hi_g, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        phi_e_hi_g = pyro.param(
            "phi_e_hi_g",
            torch.tensor(self.init_phi_e_hi_g, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        logit_p_zero_e_hi_g = pyro.param(
            "logit_p_zero_e_hi_g",
            torch.tensor(self.init_logit_p_zero_e_hi_g, device=self.device, dtype=self.dtype))

        # useful auxiliary quantities
        family_size_vector_obs_r = torch.arange(
            1, max_family_size + 1, device=self.device, dtype=self.dtype)
        family_size_vector_full_r = torch.arange(
            0, max_family_size + 1, device=self.device, dtype=self.dtype)
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)

        if not self.train_chimera_rate_params:
            alpha_c = alpha_c.detach()
            beta_c = beta_c.detach()

        # fsd xi prior distribution
        fsd_xi_prior_dist = self._get_fsd_xi_prior_dist(
            fsd_xi_prior_locs_kq,
            fsd_xi_prior_scales_kq)

        with pyro.plate("collapsed_gene_cell", size=mb_size):

            with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
                # sample gene family size distribution parameters
                fsd_xi_nq = pyro.sample("fsd_xi_nq", fsd_xi_prior_dist)

            # transform to the constrained space
            fsd_params_dict = self.fsd_codec.decode(fsd_xi_nq)

            # get chimeric and real family size distributions
            fsd_lo_dist, fsd_hi_dist = self.fsd_codec.get_fsd_components(
                fsd_params_dict,
                downsampling_rate_tensor=downsampling_rate_tensor_n)

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
            p_obs_lo_to_p_obs_hi_ratio_n = p_obs_lo_n / p_obs_hi_n
            phi_fsd_lo_comps_nj = fsd_params_dict['phi_lo']
            phi_fsd_hi_comps_nj = fsd_params_dict['phi_hi']
            mu_fsd_lo_comps_nj = fsd_params_dict['mu_lo']
            mu_fsd_hi_comps_nj = fsd_params_dict['mu_hi']
            w_fsd_lo_comps_nj = fsd_params_dict['w_lo']
            w_fsd_hi_comps_nj = fsd_params_dict['w_hi']
            mu_fsd_lo_comps_to_mu_empirical_ratio_nj = mu_fsd_lo_comps_nj / (
                self.EPS + empirical_fsd_mu_hi_tensor_n.unsqueeze(-1))
            mu_fsd_hi_comps_to_mu_empirical_ratio_nj = mu_fsd_hi_comps_nj / (
                self.EPS + empirical_fsd_mu_hi_tensor_n.unsqueeze(-1))

            # observation probability for each component of the distribution
            alpha_fsd_lo_comps_nj = (self.EPS + phi_fsd_lo_comps_nj).reciprocal()
            log_p_unobs_lo_comps_nj = alpha_fsd_lo_comps_nj * (
                    alpha_fsd_lo_comps_nj.log() - (alpha_fsd_lo_comps_nj + mu_fsd_lo_comps_nj).log())
            p_obs_lo_comps_nj = get_log_prob_compl(log_p_unobs_lo_comps_nj).exp()
            alpha_fsd_hi_comps_nj = (self.EPS + phi_fsd_hi_comps_nj).reciprocal()
            log_p_unobs_hi_comps_nj = alpha_fsd_hi_comps_nj * (
                    alpha_fsd_hi_comps_nj.log() - (alpha_fsd_hi_comps_nj + mu_fsd_hi_comps_nj).log())
            p_obs_hi_comps_nj = get_log_prob_compl(log_p_unobs_hi_comps_nj).exp()
            
            # slicing expression mu and phi by gene_index_tensor -- we only need these slices later on
            phi_e_hi_n = phi_e_hi_g[gene_index_tensor_n]
            mu_e_hi_n = mu_e_hi_g[gene_index_tensor_n]
            logit_p_zero_e_hi_n = logit_p_zero_e_hi_g[gene_index_tensor_n]

            # empirical "cell size" scale estimate
            cell_size_scale_n = total_obs_reads_per_cell_tensor_n / (
                self.median_total_reads_per_cell * downsampling_rate_tensor_n)

            # calculate p_lo and p_hi on all observable family sizes
            log_prob_fsd_lo_full_nr = fsd_lo_dist.log_prob(family_size_vector_full_r)
            log_prob_fsd_hi_full_nr = fsd_hi_dist.log_prob(family_size_vector_full_r)
            log_prob_fsd_lo_obs_nr = log_prob_fsd_lo_full_nr[..., 1:]
            log_prob_fsd_hi_obs_nr = log_prob_fsd_hi_full_nr[..., 1:]

            # calculate the (poisson) rate of chimeric molecule formation
            mu_e_lo_n = self._get_mu_e_lo_n(
                alpha_c,
                beta_c,
                cell_size_scale_n,
                downsampling_rate_tensor_n,
                logit_p_zero_e_hi_n,
                mu_e_hi_n,
                mu_fsd_hi_n,
                phi_e_hi_n)

            if posterior_sampling_mode:

                # just return the calculated auxiliary tensors
                return locals()

            else:

                # sample the fingerprint
                self._sample_fingerprint(
                    batch_shape,
                    cell_sampling_site_scale_factor_tensor_n,
                    fingerprint_tensor_nr,
                    log_prob_fsd_lo_obs_nr,
                    log_prob_fsd_hi_obs_nr,
                    mu_e_lo_n,
                    mu_e_hi_n,
                    phi_e_hi_n,
                    logit_p_zero_e_hi_n,
                    cell_size_scale_n)

                # sample fsd sparsity regularization
                if self.enable_fsd_w_dirichlet_reg:
                    self._sample_fsd_weight_sparsity_regularization(
                        fsd_params_dict,
                        gene_sampling_site_scale_factor_tensor_n)

                # sample (soft) constraints
                self._sample_gene_plate_soft_constraints(
                    locals(),
                    gene_sampling_site_scale_factor_tensor_n,
                    batch_shape)

    def _sample_fingerprint(self,
                            batch_shape: torch.Size,
                            cell_sampling_site_scale_factor_tensor_n: torch.Tensor,
                            fingerprint_tensor_nr: torch.Tensor,
                            log_prob_fsd_lo_obs_r: torch.Tensor,
                            log_prob_fsd_hi_obs_r: torch.Tensor,
                            mu_e_lo_n: torch.Tensor,
                            mu_e_hi_n: torch.Tensor,
                            phi_e_hi_n: torch.Tensor,
                            logit_p_zero_e_hi_n: torch.Tensor,
                            cell_size_scale_n: torch.Tensor):

        # calculate the fingerprint log likelihood
        fingerprint_log_likelihood_n = self._get_fingerprint_log_likelihood_monte_carlo(
            fingerprint_tensor_nr,
            log_prob_fsd_lo_obs_r,
            log_prob_fsd_hi_obs_r,
            mu_e_lo_n,
            mu_e_hi_n * cell_size_scale_n,
            phi_e_hi_n,
            logit_p_zero_e_hi_n,
            self.fingerprint_log_likelihood_n_particles)

        # sample
        with poutine.scale(scale=cell_sampling_site_scale_factor_tensor_n):
            pyro.sample("fingerprint_and_expression_observation",
                        CustomLogProbTerm(
                            custom_log_prob=fingerprint_log_likelihood_n,
                            batch_shape=batch_shape,
                            event_shape=torch.Size([])),
                        obs=torch.zeros_like(fingerprint_log_likelihood_n))

    def _get_mu_e_lo_n(self,
                       alpha_c: torch.Tensor,
                       beta_c: torch.Tensor,
                       cell_size_scale_n: torch.Tensor,
                       downsampling_rate_tensor_n: torch.Tensor,
                       logit_p_zero_e_hi_n: torch.Tensor,
                       mu_e_hi_n: torch.Tensor,
                       mu_fsd_hi_n: torch.Tensor,
                       phi_e_hi_n: torch.Tensor):
        e_hi_prior_dist_global = ZeroInflatedNegativeBinomial(
            logit_zero=logit_p_zero_e_hi_n,
            mu=mu_e_hi_n,
            phi=phi_e_hi_n)
        mean_e_hi_n = e_hi_prior_dist_global.mean
        normalized_total_fragments_n = mean_e_hi_n * mu_fsd_hi_n / (
                self.median_fsd_mu_hi * downsampling_rate_tensor_n)
        mu_e_lo_n = (alpha_c + beta_c * cell_size_scale_n) * normalized_total_fragments_n
        return mu_e_lo_n

    @staticmethod
    def _get_fingerprint_log_likelihood_monte_carlo(fingerprint_tensor_nr: torch.Tensor,
                                                    log_prob_fsd_lo_full_nr: torch.Tensor,
                                                    log_prob_fsd_hi_full_nr: torch.Tensor,
                                                    mu_e_lo_n: torch.Tensor,
                                                    mu_e_hi_n: torch.Tensor,
                                                    phi_e_hi_n: torch.Tensor,
                                                    logit_p_zero_e_hi_n: torch.Tensor,
                                                    n_particles: int) -> torch.Tensor:
        # pre-compute useful tensors
        p_lo_obs_nr = log_prob_fsd_lo_full_nr.exp()
        total_obs_rate_lo_n = mu_e_lo_n * p_lo_obs_nr.sum(-1)
        log_rate_e_lo_nr = mu_e_lo_n.log().unsqueeze(-1) + log_prob_fsd_lo_full_nr

        p_hi_obs_nr = log_prob_fsd_hi_full_nr.exp()
        total_obs_rate_hi_n = mu_e_hi_n * p_hi_obs_nr.sum(-1)
        log_rate_e_hi_nr = mu_e_hi_n.log().unsqueeze(-1) + log_prob_fsd_hi_full_nr

        # fingerprint_log_norm_factor_n = (fingerprint_tensor_nr + 1).lgamma().sum(-1)

        log_p_zero_e_hi_n = torch.nn.functional.logsigmoid(logit_p_zero_e_hi_n)
        log_p_nonzero_e_hi_n = get_log_prob_compl(log_p_zero_e_hi_n)

        # reparameterized Monte-Carlo samples from Gamma(alpha, alpha) for approximate e_hi marginalization
        alpha_e_hi_n = phi_e_hi_n.reciprocal()
        omega_mn = dist.Gamma(concentration=alpha_e_hi_n, rate=alpha_e_hi_n).rsample((n_particles,))

        # contribution of chimeric molecules alone
        log_poisson_zero_e_hi_contrib_n = (
                log_p_zero_e_hi_n
                + (fingerprint_tensor_nr * log_rate_e_lo_nr).sum(-1)
                - total_obs_rate_lo_n)
                # - fingerprint_log_norm_factor_n)

        # log combined (chimeric and real) Poisson rate for each Gamma particle
        log_rate_combined_mnr = logaddexp(
            log_rate_e_lo_nr,
            log_rate_e_hi_nr + omega_mn.log().unsqueeze(-1))
        log_poisson_nonzero_e_hi_contrib_mn = (
            (fingerprint_tensor_nr * log_rate_combined_mnr).sum(-1)
            - (total_obs_rate_lo_n + total_obs_rate_hi_n * omega_mn))
            #- fingerprint_log_norm_factor_n)
        log_poisson_nonzero_e_hi_contrib_n = (
            log_poisson_nonzero_e_hi_contrib_mn.logsumexp(0)
            - np.log(n_particles)
            + log_p_nonzero_e_hi_n)

        log_like_n = logaddexp(
            log_poisson_zero_e_hi_contrib_n,
            log_poisson_nonzero_e_hi_contrib_n)

        return log_like_n

    def _get_fsd_xi_prior_dist(self,
                               fsd_xi_prior_locs_kq: torch.Tensor,
                               fsd_xi_prior_scales_kq: torch.Tensor):
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

    def _sample_gene_plate_soft_constraints(self, model_vars_dict, scale_factor_tensor, batch_shape):
        with poutine.scale(scale=scale_factor_tensor):
            for var_name, var_constraint_params in self.model_constraint_params_dict.items():
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

    def _sample_fsd_weight_sparsity_regularization(self, fsd_params_dict, scale_factor_tensor):
        with poutine.scale(scale=scale_factor_tensor):
            if self.fsd_codec.n_fsd_lo_comps > 1:
                with poutine.scale(scale=self.w_lo_dirichlet_reg_strength):
                    pyro.sample(
                        "w_lo_dirichlet_reg",
                        dist.Dirichlet(
                            self.w_lo_dirichlet_concentration * torch.ones_like(fsd_params_dict['w_lo'])),
                        obs=fsd_params_dict['w_lo'])
            if self.fsd_codec.n_fsd_hi_comps > 1:
                with poutine.scale(scale=self.w_hi_dirichlet_reg_strength):
                    pyro.sample(
                        "w_hi_dirichlet_reg",
                        dist.Dirichlet(
                            self.w_hi_dirichlet_concentration * torch.ones_like(fsd_params_dict['w_hi'])),
                        obs=fsd_params_dict['w_hi'])

    def guide(self,
              data: Dict[str, torch.Tensor],
              posterior_sampling_mode: bool = False):

        # input tensors
        fingerprint_tensor_nr = data['fingerprint_tensor']
        gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
        gene_index_tensor_n = data['gene_index_tensor']

        # sizes
        mb_size = fingerprint_tensor_nr.shape[0]

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

        # point estimate for fsd_xi (gene)
        fsd_xi_posterior_loc_gq = pyro.param(
            "fsd_xi_posterior_loc_gq",
            self.fsd_codec.get_sorted_fsd_xi(self.fsd_codec.init_fsd_xi_loc_posterior))
        
        # base posterior distribution for xi
        if self.guide_type == 'map':
            fsd_xi_posterior_base_dist = dist.Delta(
                v=fsd_xi_posterior_loc_gq[gene_index_tensor_n, :]).to_event(1)
        elif self.guide_type == 'gaussian':
            fsd_xi_posterior_scale_gq = pyro.param(
                "fsd_xi_posterior_scale_gq",
                self.fsd_gmm_init_xi_scale * torch.ones(
                    (self.n_total_genes, self.fsd_codec.total_fsd_params), device=self.device, dtype=self.dtype),
                constraint=constraints.greater_than(self.fsd_xi_posterior_min_scale))
            fsd_xi_posterior_base_dist = dist.Normal(
                loc=fsd_xi_posterior_loc_gq[gene_index_tensor_n, :],
                scale=fsd_xi_posterior_scale_gq[gene_index_tensor_n, :]).to_event(1)
        else:
            raise Exception("Unknown guide_type!")
        
        # apply a pseudo-bijective transformation to sort xi by component weights
        fsd_xi_sort_trans = SortByComponentWeights(self.fsd_codec)
        fsd_xi_posterior_dist = dist.TransformedDistribution(
            fsd_xi_posterior_base_dist, [fsd_xi_sort_trans])
        
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
                 sc_family_size_model: SingleCellFamilySizeModel,
                 device: torch.device,
                 dtype: torch.dtype):
        """Initializer.

        :param sc_family_size_model: an instance of ``SingleCellFamilySizeModel``
        :param device: torch device
        :param dtype: torch dtype
        """
        self.sc_family_size_model = sc_family_size_model
        self.device = device
        self.dtype = dtype

    def _generate_single_gene_minibatch_data(self,
                                             gene_index: int,
                                             i_cell_begin: int,
                                             i_cell_end: int,
                                             n_particles_cell: int) -> Dict[str, torch.Tensor]:
        """Generate model input tensors for a given gene index and the cell index range

        :param n_particles_cell: repeat factor for every cell

        .. note: The generated minibatch has scale-factor set to 1.0 for all gene and cell sampling
            sites (because they are not necessary for our purposes here). As such, the minibatches
            produced by this method should not be used for training.
        """
        cell_index_array = np.repeat(np.arange(i_cell_begin, i_cell_end, dtype=np.int), n_particles_cell)
        gene_index_array = gene_index * np.ones_like(cell_index_array)
        cell_sampling_site_scale_factor_array = np.ones_like(cell_index_array)
        gene_sampling_site_scale_factor_array = np.ones_like(cell_index_array)

        return self.sc_family_size_model.sc_fingerprint_datastore.generate_torch_minibatch_data(
            cell_index_array,
            gene_index_array,
            cell_sampling_site_scale_factor_array,
            gene_sampling_site_scale_factor_array,
            self.device,
            self.dtype)

    @torch.no_grad()
    def _get_trained_model_context(self, minibatch_data: Dict[str, torch.Tensor]) \
            -> Dict[str, Any]:
        """Samples the posterior on a given minibatch, replays it on the model, and returns a
        dictionary of intermediate tensors that appear in the model evaluated at the posterior
        samples."""
        guide_trace = poutine.trace(self.sc_family_size_model.guide).get_trace(
            minibatch_data, posterior_sampling_mode=True)
        trained_model = poutine.replay(self.sc_family_size_model.model, trace=guide_trace)
        trained_model_trace = poutine.trace(trained_model).get_trace(
            minibatch_data, posterior_sampling_mode=True)
        return trained_model_trace.nodes["_RETURN"]["value"]

    @torch.no_grad()
    def _generate_omega_importance_sampler_inputs(
            self,
            gene_index: int,
            i_cell_begin: int,
            i_cell_end: int,
            n_particles_cell: int,
            run_mode: str) -> Tuple[PosteriorImportanceSamplerInputs, Dict[str, Any], Dict[str, Any]]:
        """Generates the required inputs for ``PosteriorImportanceSampler`` to calculate the mean
        and variance of gene expression, assuming no zero-inflation of the prior for :math:`\mu^>`.

        .. note:: According to the model, the prior for :math:`\mu^<`, the Poisson rate for chimeric
            molecules, is deterministic; as such, it is directly picked up from the trained model
            context ``trained_model_context["mu_e_lo_n"]`` and no marginalization is necessary.

            The prior :math:`\mu^>`, the Poisson rate for real molecules, however, is a zero-inflated
            Gamma and must be marginalized. This method generates importance sampling inputs for
            marginalizing :math:`\mu^>` and calculing the first and second moments of of gene expression,
            for a non-zero-inflated Gamma. We parametrize :math:`\mu^>` as follows:

            .. math::

                \mu^> | no-zero-inflation = \mathbb{E}[\mu^>] \omega^>,

                \omega^> \sim \Gamma(\alpha^>, \alpha^>),

                \alpha^> = 1 / \phi^>,

            where :math:`\phi^>` is the prior over-dipersion of expression and :math:`\mathbb{E}[\mu^>]`
            is the prior mean of the expression.

        :param gene_index: index of the gene in the datastore
        :param run_mode: choices include ``only_observed`` and ``full``. In the only_observed mode,
            we only calculate how many of the observed molecules are real (as opposed to background/chimeric).
            In the ``full`` mode, we also include currently unobserved molecules; that is, we estimate
            how many real molecules we expect to observe in the limit of infinite sequencing depth.
        :param i_cell_begin: begin cell index (inclusive)
        :param i_cell_end: end cell index (exclusive)
        :param n_particles_cell: how many repeats per cell
        :return: a tuple that matches the *args signature of ``PosteriorImportanceSampler.__init__``, and
            the trained model context.
        """
        assert run_mode in {"only_observed", "full"}
        minibatch_data = self._generate_single_gene_minibatch_data(
            gene_index, i_cell_begin, i_cell_end, n_particles_cell)
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
            """Calculates the log likelihood of the fingerprint for given :math:`\omega^>` proposals.

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
            """Calculates the mean and the variance of gene expression over the :math:`\omega^>` proposals
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

        # Gamma concentration and rates for prior and proposal distributions of :math:`\omega`
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
            i_cell_begin: int,
            i_cell_end: int,
            n_particles_omega: int,
            output_ess: bool,
            run_mode: str):
        """

        :param run_mode: see ``_generate_omega_importance_sampler_inputs``
        :param gene_index: index of the gene in the datastore
        :param i_cell_begin: begin cell index (inclusive)
        :param i_cell_end: end cell index (exclusive)
        :param n_particles_omega: number of random proposals used for importance sampling
        :return: estimated first and second moments of gene expression; if output_ess == True, also the ESS of
            the two estimators, in order.
        """

        # generate inputs for importance sampling with no zero inflation and the posterior model context
        omega_importance_sampler_inputs, trained_model_context, _ = self._generate_omega_importance_sampler_inputs(
            gene_index=gene_index,
            i_cell_begin=i_cell_begin,
            i_cell_end=i_cell_end,
            n_particles_cell=1,
            run_mode=run_mode)

        # perform importance sampling assuming no zero inflation
        batch_size = i_cell_end - i_cell_begin
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

    def get_gene_expression_moments_summary(
            self,
            gene_index: int,
            n_particles_omega: int,
            n_particles_xi: int,
            cell_shard_size: int,
            run_mode: str = 'full',
            mode_estimation_strategy: str = 'lower_bound') -> Dict[str, np.ndarray]:
        """Calculate posterior gene expression summary/

        :param gene_index: index of the gene in the datastore
        :param n_particles_omega: number of random proposals used for importance sampling of :math:`\omega^>`
        :param n_particles_xi: number of posterior samples from the guide
        :param cell_shard_size: how many cells to include in every batch
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
            def __get_mode(mean: torch.Tensor, std: torch.Tensor):
                return torch.clamp(torch.ceil(mean - np.sqrt(3) * std), 0)
        elif mode_estimation_strategy == 'upper_bound':
            def __get_mode(mean: torch.Tensor, std: torch.Tensor):
                return torch.clamp(torch.floor(mean + np.sqrt(3) * std), 0)
        else:
            raise ValueError("Invalid mode_estimation_strategy; valid options are `lower_bound` and `upper_bound`")

        e_hi_mean_shard_list = []
        e_hi_std_shard_list = []
        e_hi_mode_shard_list = []

        n_cells = self.sc_family_size_model.sc_fingerprint_datastore.n_cells
        for cell_shard in range(n_cells // cell_shard_size + 1):
            i_cell_begin = min(cell_shard * cell_shard_size, n_cells)
            i_cell_end = min((cell_shard + 1) * cell_shard_size, n_cells)
            if i_cell_begin == i_cell_end:
                break

            log_e_hi_mom_1_n_list = []
            log_e_hi_mom_2_n_list = []
            for _ in range(n_particles_xi):
                log_e_hi_mom_1_n, log_e_hi_mom_2_n = self._estimate_log_gene_expression_with_fixed_n_particles(
                    gene_index=gene_index,
                    i_cell_begin=i_cell_begin,
                    i_cell_end=i_cell_end,
                    n_particles_omega=n_particles_omega,
                    output_ess=False,
                    run_mode=run_mode)
                log_e_hi_mom_1_n_list.append(log_e_hi_mom_1_n)
                log_e_hi_mom_2_n_list.append(log_e_hi_mom_2_n)
            log_e_hi_mom_1_n = torch.logsumexp(torch.stack(log_e_hi_mom_1_n_list), 0) - np.log(n_particles_xi)
            log_e_hi_mom_2_n = torch.logsumexp(torch.stack(log_e_hi_mom_2_n_list), 0) - np.log(n_particles_xi)

            mean_e_hi_n = log_e_hi_mom_1_n.exp()
            std_e_hi_n = torch.clamp(log_e_hi_mom_2_n.exp() - mean_e_hi_n.pow(2), 0).sqrt()
            mode_e_hi_n = __get_mode(mean_e_hi_n, std_e_hi_n)

            e_hi_mean_shard_list.append(__fix_scalar(mean_e_hi_n.cpu().numpy()))
            e_hi_std_shard_list.append(__fix_scalar(std_e_hi_n.cpu().numpy()))
            e_hi_mode_shard_list.append(__fix_scalar(mode_e_hi_n.cpu().numpy()))

        return {'e_hi_map': np.concatenate(e_hi_mode_shard_list),
                'e_hi_std': np.concatenate(e_hi_std_shard_list),
                'e_hi_mean': np.concatenate(e_hi_mean_shard_list)}

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

        # calculate the log norm factor of the proper part of :math:`\omega` posterior
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
            i_cell_begin: int,
            i_cell_end: int,
            n_proposals_omega: int,
            n_particles_omega: int,
            n_particles_cell: int,
            n_particles_expression: int,
            run_mode: str):
        """

        .. note:: "proper" means non-zero-inflated part throughput.

        :param gene_index:
        :param i_cell_begin:
        :param i_cell_end:
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
            i_cell_begin=i_cell_begin,
            i_cell_end=i_cell_end,
            n_particles_cell=n_particles_cell,
            run_mode=run_mode)
        batch_size = n_particles_cell * (i_cell_end - i_cell_begin)

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

        # draw gene expression for each omega particle
        log_prob_fsd_hi_full_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_full_nr"]
        log_prob_fsd_lo_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_lo_obs_nr"]
        log_prob_fsd_hi_obs_nr: torch.Tensor = trained_model_context["log_prob_fsd_hi_obs_nr"]
        mu_e_lo_n: torch.Tensor = trained_model_context["mu_e_lo_n"]
        mu_e_hi_n: torch.Tensor = trained_model_context["mu_e_hi_n"]
        fingerprint_tensor_nr: torch.Tensor = minibatch_data["fingerprint_tensor"]
        log_mu_e_lo_n = mu_e_lo_n.log()
        log_mu_e_hi_n = mu_e_hi_n.log()
        log_rate_e_lo_nr = log_mu_e_lo_n.unsqueeze(-1) + log_prob_fsd_lo_obs_nr
        log_rate_e_hi_nr = log_mu_e_hi_n.unsqueeze(-1) + log_prob_fsd_hi_obs_nr
        log_omega_mn = omega_posterior_samples_mn.log()
        log_rate_e_hi_mnr = log_rate_e_hi_nr + log_omega_mn.unsqueeze(-1)
        log_rate_combined_mnr = logaddexp(log_rate_e_lo_nr, log_rate_e_hi_mnr)
        log_p_binom_obs_hi_mnr = log_rate_e_hi_mnr - log_rate_combined_mnr
        logit_p_binom_obs_hi_mnr = log_p_binom_obs_hi_mnr - get_log_prob_compl(log_p_binom_obs_hi_mnr)

        p_obs_hi_n = log_prob_fsd_hi_full_nr[:, 0].exp()
        e_hi_unobs_dist = torch.distributions.Poisson(
            rate=p_obs_hi_n * omega_posterior_samples_mn * mu_e_hi_n)
        e_hi_obs_dist = torch.distributions.Binomial(
            total_count=fingerprint_tensor_nr,
            logits=logit_p_binom_obs_hi_mnr)

        e_hi_unobs_samples_smn = e_hi_unobs_dist.sample((n_particles_expression,))
        e_hi_obs_samples_smnr = e_hi_obs_dist.sample((n_particles_expression,))

        return e_hi_unobs_samples_smn, e_hi_obs_samples_smnr
