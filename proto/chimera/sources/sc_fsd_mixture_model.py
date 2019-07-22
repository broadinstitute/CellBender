import numpy as np
from typing import Tuple, List, Dict, Union

import pyro
from pyro import poutine
from pyro.infer import Trace_ELBO
import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution

import torch
from torch.distributions import constraints, transforms
from torch.distributions.utils import broadcast_all
from torch.distributions.transforms import Transform
from torch.nn.parameter import Parameter

from pyro_extras import CustomLogProbTerm, NegativeBinomial, ZeroInflatedNegativeBinomial, \
    MixtureDistribution, get_hellinger_distance, logit, logaddexp, get_log_prob_compl, \
    get_confidence_interval
from sc_fingerprint import SingleCellFingerprintDataStore, \
    generate_downsampled_minibatch

import logging
from abc import abstractmethod
from collections import defaultdict


class FamilySizeDistributionCodec(torch.nn.Module):
    def __init__(self):
        super(FamilySizeDistributionCodec, self).__init__()

    @property
    @abstractmethod
    def total_fsd_params(self):
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, fsd_params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, fsd_xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        
    @abstractmethod
    def get_sorted_fsd_xi(self, fsd_xi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
                
    @abstractmethod
    def get_fsd_components(self,
                           fsd_params_dict: Dict[str, torch.Tensor],
                           downsampling_rate_tensor: Union[None, torch.Tensor]) \
            -> Tuple[TorchDistribution, TorchDistribution]:
        raise NotImplementedError

    @property
    @abstractmethod
    def init_fsd_xi_loc_prior(self) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def init_fsd_xi_loc_posterior(self) -> torch.Tensor:
        raise NotImplementedError
        

class GeneralNegativeBinomialMixtureFamilySizeDistributionCodec(FamilySizeDistributionCodec):
    def __init__(self,
                 sc_fingerprint_datastore: SingleCellFingerprintDataStore,
                 n_fsd_lo_comps: int,
                 n_fsd_hi_comps: int,
                 fsd_init_params_dict: Dict[str, float],
                 device=torch.device("cuda"),
                 dtype=torch.float):
        super(GeneralNegativeBinomialMixtureFamilySizeDistributionCodec, self).__init__()
        self.sc_fingerprint_datastore = sc_fingerprint_datastore
        self.n_fsd_lo_comps = n_fsd_lo_comps
        self.n_fsd_hi_comps = n_fsd_hi_comps
        self.fsd_init_params_dict = fsd_init_params_dict
        
        self.fsd_init_min_mu_lo = fsd_init_params_dict['fsd_init.min_mu_lo']
        self.fsd_init_min_mu_hi = fsd_init_params_dict['fsd_init.min_mu_hi']
        self.fsd_init_max_phi_lo = fsd_init_params_dict['fsd_init.max_phi_lo']
        self.fsd_init_max_phi_hi = fsd_init_params_dict['fsd_init.max_phi_hi']
        self.fsd_init_mu_decay = fsd_init_params_dict['fsd_init.mu_decay']
        self.fsd_init_w_decay = fsd_init_params_dict['fsd_init.w_decay']
        self.fsd_init_mu_lo_to_mu_hi_ratio = fsd_init_params_dict['fsd_init.mu_lo_to_mu_hi_ratio']

        self.device = device
        self.dtype = dtype
        self.stick = transforms.StickBreakingTransform()
        
        # initialization of p_lo and p_hi
        mean_fsd_mu_hi = np.mean(sc_fingerprint_datastore.empirical_fsd_mu_hi)
        mean_fsd_phi_hi = np.mean(sc_fingerprint_datastore.empirical_fsd_phi_hi)
        (self.init_fsd_mu_lo, self.init_fsd_phi_lo, self.init_fsd_w_lo,
         self.init_fsd_mu_hi, self.init_fsd_phi_hi, self.init_fsd_w_hi) = self.generate_fsd_init_params(
            mean_fsd_mu_hi, mean_fsd_phi_hi)
        
        # caches
        self._init_fsd_xi_loc_posterior = None

    @property
    def total_fsd_params(self):
        n_lo = 3 * self.n_fsd_lo_comps - 1
        n_hi = 3 * self.n_fsd_hi_comps - 1
        return n_lo + n_hi

    def decode(self, fsd_xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert fsd_xi.shape[-1] == self.total_fsd_params
        offset = 0

        # p_hi parameters are directly transformed from fsd_xi
        log_mu_hi = fsd_xi[..., offset:(offset + self.n_fsd_hi_comps)]
        mu_hi = log_mu_hi.exp()
        offset += self.n_fsd_hi_comps
        
        log_phi_hi = fsd_xi[..., offset:(offset + self.n_fsd_hi_comps)]
        phi_hi = log_phi_hi.exp()
        offset += self.n_fsd_hi_comps
        
        if self.n_fsd_hi_comps > 1:
            w_hi = self.stick(fsd_xi[..., offset:(offset + self.n_fsd_hi_comps - 1)])
            offset += (self.n_fsd_hi_comps - 1)
        else:
            w_hi = torch.ones_like(mu_hi)

        # p_lo parameters are directly transformed from fsd_xi
        log_mu_lo = fsd_xi[..., offset:(offset + self.n_fsd_lo_comps)]
        mu_lo = log_mu_lo.exp()
        offset += self.n_fsd_lo_comps
        
        log_phi_lo = fsd_xi[..., offset:(offset + self.n_fsd_lo_comps)]
        phi_lo = log_phi_lo.exp()
        offset += self.n_fsd_lo_comps
        
        if self.n_fsd_lo_comps > 1:
            w_lo = self.stick(fsd_xi[..., offset:(offset + self.n_fsd_lo_comps - 1)])
            offset += (self.n_fsd_lo_comps - 1)
        else:
            w_lo = torch.ones_like(mu_lo)
        
        return {'mu_lo': mu_lo,
                'phi_lo': phi_lo,
                'w_lo': w_lo,
                'mu_hi': mu_hi,
                'phi_hi': phi_hi,
                'w_hi': w_hi}
    
    def encode(self, fsd_params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        xi_tuple = tuple()
        xi_tuple += (fsd_params_dict['mu_hi'].log(),)
        xi_tuple += (fsd_params_dict['phi_hi'].log(),)
        if self.n_fsd_hi_comps > 1:
            xi_tuple += (self.stick.inv(fsd_params_dict['w_hi']),)
        xi_tuple += (fsd_params_dict['mu_lo'].log(),)
        xi_tuple += (fsd_params_dict['phi_lo'].log(),)
        if self.n_fsd_lo_comps > 1:
            xi_tuple += (self.stick.inv(fsd_params_dict['w_lo']),)
        return torch.cat(xi_tuple, -1)
    
    def get_sorted_params_dict(self, fsd_params_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fsd_lo_sort_order = torch.argsort(fsd_params_dict['w_lo'], dim=-1, descending=True)
        fsd_hi_sort_order = torch.argsort(fsd_params_dict['w_hi'], dim=-1, descending=True)
        sorted_fsd_params_dict = {
            'mu_lo': torch.gather(fsd_params_dict['mu_lo'], dim=-1, index=fsd_lo_sort_order),
            'phi_lo': torch.gather(fsd_params_dict['phi_lo'], dim=-1, index=fsd_lo_sort_order),
            'w_lo': torch.gather(fsd_params_dict['w_lo'], dim=-1, index=fsd_lo_sort_order),
            'mu_hi': torch.gather(fsd_params_dict['mu_hi'], dim=-1, index=fsd_hi_sort_order),
            'phi_hi': torch.gather(fsd_params_dict['phi_hi'], dim=-1, index=fsd_hi_sort_order),
            'w_hi': torch.gather(fsd_params_dict['w_hi'], dim=-1, index=fsd_hi_sort_order)}
        return sorted_fsd_params_dict

    # TODO
    # this needs to be rewritten using direct ops
    # the current implementation is not optimal
    def get_sorted_fsd_xi(self, fsd_xi: torch.Tensor) -> torch.Tensor:
        return self.encode(self.get_sorted_params_dict(self.decode(fsd_xi)))

    def get_fsd_components(self,
                           fsd_params_dict: Dict[str, torch.Tensor],
                           downsampling_rate_tensor: Union[None, torch.Tensor]) \
            -> Tuple[TorchDistribution, TorchDistribution]:
        # instantiate the "chimeric" (lo) distribution
        log_w_nb_lo_tuple = tuple(fsd_params_dict['w_lo'][..., j].log().unsqueeze(-1) for j in range(self.n_fsd_lo_comps))
        if downsampling_rate_tensor is None:
            nb_lo_components_tuple = tuple(NegativeBinomial(
                fsd_params_dict['mu_lo'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_lo'][..., j].unsqueeze(-1)) for j in range(self.n_fsd_lo_comps))
        else:
            nb_lo_components_tuple = tuple(NegativeBinomial(
                downsampling_rate_tensor.unsqueeze(-1) * fsd_params_dict['mu_lo'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_lo'][..., j].unsqueeze(-1)) for j in range(self.n_fsd_lo_comps))

        # instantiate the "real" (hi) distribution
        log_w_nb_hi_tuple = tuple(fsd_params_dict['w_hi'][..., j].log().unsqueeze(-1) for j in range(self.n_fsd_hi_comps))
        if downsampling_rate_tensor is None:
            nb_hi_components_tuple = tuple(NegativeBinomial(
                fsd_params_dict['mu_hi'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_hi'][..., j].unsqueeze(-1)) for j in range(self.n_fsd_hi_comps))
        else:
            nb_hi_components_tuple = tuple(NegativeBinomial(
                downsampling_rate_tensor.unsqueeze(-1) * fsd_params_dict['mu_hi'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_hi'][..., j].unsqueeze(-1)) for j in range(self.n_fsd_hi_comps))
            
        dist_lo = MixtureDistribution(log_w_nb_lo_tuple, nb_lo_components_tuple)
        dist_hi = MixtureDistribution(log_w_nb_hi_tuple, nb_hi_components_tuple)

        return dist_lo, dist_hi

    def generate_fsd_init_params(self, mu_hi_guess, phi_hi_guess):
        mu_lo = self.fsd_init_mu_lo_to_mu_hi_ratio * mu_hi_guess * np.power(
            np.asarray([self.fsd_init_mu_decay]), np.arange(self.n_fsd_lo_comps))
        mu_lo = np.maximum(mu_lo, 1.1 * self.fsd_init_min_mu_lo)
        phi_lo = min(1.0, 0.9 * self.fsd_init_max_phi_lo) * np.ones((self.n_fsd_lo_comps,))
        w_lo = np.power(np.asarray([self.fsd_init_w_decay]), np.arange(self.n_fsd_lo_comps))
        w_lo = w_lo / np.sum(w_lo)

        mu_hi = mu_hi_guess * np.power(
            np.asarray([self.fsd_init_mu_decay]), np.arange(self.n_fsd_hi_comps))
        mu_hi = np.maximum(mu_hi, 1.1 * self.fsd_init_min_mu_hi)
        phi_hi = min(phi_hi_guess, 0.9 * self.fsd_init_max_phi_hi) * np.ones((self.n_fsd_hi_comps,))
        w_hi = np.power(np.asarray([self.fsd_init_w_decay]), np.arange(self.n_fsd_hi_comps))
        w_hi = w_hi / np.sum(w_hi)
        
        return mu_lo, phi_lo, w_lo, mu_hi, phi_hi, w_hi

    @property
    def init_fsd_xi_loc_prior(self):
        mu_lo = torch.tensor(self.init_fsd_mu_lo, device=self.device, dtype=self.dtype)
        phi_lo = torch.tensor(self.init_fsd_phi_lo, device=self.device, dtype=self.dtype)
        w_lo = torch.tensor(self.init_fsd_w_lo, device=self.device, dtype=self.dtype)
        mu_hi = torch.tensor(self.init_fsd_mu_hi, device=self.device, dtype=self.dtype)
        phi_hi = torch.tensor(self.init_fsd_phi_hi, device=self.device, dtype=self.dtype)
        w_hi = torch.tensor(self.init_fsd_w_hi, device=self.device, dtype=self.dtype)
        
        return self.encode({
            'mu_lo': mu_lo,
            'phi_lo': phi_lo,
            'w_lo': w_lo,
            'mu_hi': mu_hi,
            'phi_hi': phi_hi,
            'w_hi': w_hi})
    
    @property
    def init_fsd_xi_loc_posterior(self):
        if self._init_fsd_xi_loc_posterior is None:
            xi_list = []
            for i_gene in range(self.sc_fingerprint_datastore.n_genes):
                mu_lo, phi_lo, w_lo, mu_hi, phi_hi, w_hi = self.generate_fsd_init_params(
                    self.sc_fingerprint_datastore.empirical_fsd_mu_hi[i_gene],
                    self.sc_fingerprint_datastore.empirical_fsd_phi_hi[i_gene])
                xi = self.encode({
                    'mu_lo': torch.tensor(mu_lo, dtype=self.dtype),
                    'phi_lo': torch.tensor(phi_lo, dtype=self.dtype),
                    'w_lo': torch.tensor(w_lo, dtype=self.dtype),
                    'mu_hi': torch.tensor(mu_hi, dtype=self.dtype),
                    'phi_hi': torch.tensor(phi_hi, dtype=self.dtype),
                    'w_hi': torch.tensor(w_hi, dtype=self.dtype)})
                xi_list.append(xi.unsqueeze(0))
            self._init_fsd_xi_loc_posterior = torch.cat(xi_list, 0).to(self.device)
        return self._init_fsd_xi_loc_posterior



                    
class SortByComponentWeights(Transform):
    def __init__(self, fsd_codec: FamilySizeDistributionCodec):
        super(SortByComponentWeights, self).__init__()
        self.fsd_codec = fsd_codec
        self._intermediates_cache = {}

    def _call(self, x):
        y = self.fsd_codec.get_sorted_fsd_xi(x)
        self._add_intermediate_to_cache(x, y)
        return y
    
    def _inverse(self, y):
        if y in self._intermediates_cache:
            x = self._intermediates_cache.pop(y)
            return x
        else:
            raise KeyError("SortByComponentWeights expected to find "
                           "key in intermediates cache but didn't")

    def _add_intermediate_to_cache(self, x, y):
        assert(y not in self._intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[y] = x

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)

    
class SingleCellFamilySizeModel(torch.nn.Module):
    DEFAULT_E_LO_SUM_WIDTH = 10
    DEFAULT_E_HI_SUM_WIDTH = 20
    DEFAULT_CONFIDENCE_INTERVAL_LOWER = 0.05 
    DEFAULT_CONFIDENCE_INTERVAL_UPPER = 0.95
    EPS = 1e-6
    
    def __init__(self,
                 init_params_dict: dict,
                 model_constraint_params_dict: dict,
                 sc_fingerprint_datastore: SingleCellFingerprintDataStore,
                 fsd_codec: FamilySizeDistributionCodec,
                 e_lo_prior_dist: str = 'poisson',
                 e_hi_prior_dist: str = 'negbinom',
                 model_type: str = 'approx_multinomial',
                 guide_type: str = 'map',
                 device=torch.device('cuda'),
                 dtype=torch.float):
        super(SingleCellFamilySizeModel, self).__init__()

        self.model_constraint_params_dict = model_constraint_params_dict
        self.sc_fingerprint_datastore = sc_fingerprint_datastore
        self.fsd_codec = fsd_codec
        
        self.n_total_cells = sc_fingerprint_datastore.n_cells
        self.n_total_genes = sc_fingerprint_datastore.n_genes
        
        self.e_lo_prior_dist = e_lo_prior_dist
        self.e_hi_prior_dist = e_hi_prior_dist

        self.guide_type = guide_type
        self.model_type = model_type

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
        self.enable_fsd_gmm_scale_optimization = init_params_dict['fsd.enable_gmm_scale_optimization']
        self.w_lo_dirichlet_reg_strength = init_params_dict['fsd.w_lo_dirichlet_reg_strength']
        self.w_hi_dirichlet_reg_strength = init_params_dict['fsd.w_hi_dirichlet_reg_strength']
        self.w_lo_dirichlet_concentration = init_params_dict['fsd.w_lo_dirichlet_concentration']
        self.w_hi_dirichlet_concentration = init_params_dict['fsd.w_hi_dirichlet_concentration']
        self.train_chimera_rate_params = init_params_dict['chimera.enable_hyperparameter_optimization']
        self.fsd_xi_posterior_min_scale = init_params_dict['fsd.xi_posterior_min_scale']
        
        # some useful auxiliary quantities and handles
        self.median_total_reads_per_cell = np.median(sc_fingerprint_datastore.total_obs_reads_per_cell)
        self.median_fsd_mu_hi = np.median(sc_fingerprint_datastore.empirical_fsd_mu_hi)
        
        self.init_mu_e_hi = sc_fingerprint_datastore.empirical_mu_e_hi
        self.init_phi_e_hi = sc_fingerprint_datastore.empirical_phi_e_hi
        self.init_logit_p_zero_e_hi = logit(torch.tensor(sc_fingerprint_datastore.empirical_p_zero_e_hi)).numpy()
        
        self.init_phi_e_lo = init_params_dict['expr.phi_e_lo'] * np.ones_like(self.init_phi_e_hi)
        self.init_alpha_c = init_params_dict['chimera.alpha_c']
        self.init_beta_c = init_params_dict['chimera.beta_c']
        
        # logging
        self._logger = logging.getLogger()
                
    def model(self,
              data,
              calculate_expression_map=False,
              calculate_joint_expression_log_posterior=False):
        # register the parameters of the family size distribution codec
        pyro.module("fsd_codec", self.fsd_codec)
        
        # GMM prior for family size distribution parameters
        fsd_xi_prior_locs = pyro.param(
            "fsd_xi_prior_locs",
            self.fsd_codec.init_fsd_xi_loc_prior +
            self.fsd_gmm_init_components_perplexity * torch.randn(
                (self.fsd_gmm_num_components, self.fsd_codec.total_fsd_params),
                dtype=self.dtype, device=self.device))

        fsd_xi_prior_scales = pyro.param(
            "fsd_xi_prior_scales",
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
        phi_e_lo = pyro.param(
            "phi_e_lo",
            torch.tensor(self.init_phi_e_lo, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
            
        # gene expression parameters
        mu_e_hi = pyro.param(
            "mu_e_hi",
            torch.tensor(self.init_mu_e_hi, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        phi_e_hi = pyro.param(
            "phi_e_hi",
            torch.tensor(self.init_phi_e_hi, device=self.device, dtype=self.dtype),
            constraint=constraints.positive)
        logit_p_zero_e_hi = pyro.param(
            "logit_p_zero_e_hi",
            torch.tensor(self.init_logit_p_zero_e_hi, device=self.device, dtype=self.dtype))

        # sizes
        mb_size = data['fingerprint_tensor'].shape[0]
        max_family_size = data['fingerprint_tensor'].shape[1]
        
        # useful auxiliary quantities
        family_size_vector_observable = torch.arange(1, max_family_size + 1, device=self.device).type(self.dtype)
        family_size_vector_full = torch.arange(0, max_family_size + 1, device=self.device).type(self.dtype)
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        
        # expression marginalization window width around the guessed point
        if 'e_lo_sum_width' in data:
            e_lo_sum_width = data['e_lo_sum_width']
        else:
            self._logger.warning(f"'e_lo_sum_width' is not given -- fallback to default value {self.DEFAULT_E_LO_SUM_WIDTH}.")
            e_lo_sum_width = self.DEFAULT_E_LO_SUM_WIDTH
            
        if 'e_hi_sum_width' in data:
            e_hi_sum_width = data['e_hi_sum_width']
        else:
            self._logger.warning(f"'e_hi_sum_width' is not given -- fallback to default value {self.DEFAULT_E_HI_SUM_WIDTH}.")
            e_hi_sum_width = self.DEFAULT_E_HI_SUM_WIDTH

        e_lo_range = torch.arange(0, e_lo_sum_width, device=self.device).type(self.dtype)
        e_hi_range = torch.arange(0, e_hi_sum_width, device=self.device).type(self.dtype)
        
        if not self.train_chimera_rate_params:
            alpha_c = alpha_c.clone().detach()
            beta_c = beta_c.clone().detach()
            phi_e_lo = phi_e_lo.clone().detach()
            
        if not self.enable_fsd_gmm_scale_optimization:
            fsd_xi_prior_scales = fsd_xi_prior_scales.clone().detach()
            
        # global latent variables
        if self.fsd_gmm_num_components > 1:
            # generate the marginalized GMM distribution
            fsd_xi_prior_weights = pyro.sample(
                "fsd_xi_prior_weights",
                dist.Dirichlet(
                    self.fsd_gmm_dirichlet_concentration *
                    torch.ones((self.fsd_gmm_num_components,), dtype=self.dtype, device=self.device)))
            fsd_xi_prior_log_weights = fsd_xi_prior_weights.log()
            fsd_xi_prior_log_weights_tuple = tuple(
                fsd_xi_prior_log_weights[j]
                for j in range(self.fsd_gmm_num_components))        
            fsd_xi_prior_components_tuple = tuple(
                dist.Normal(fsd_xi_prior_locs[j, :], fsd_xi_prior_scales[j, :]).to_event(1)
                for j in range(self.fsd_gmm_num_components))
            fsd_xi_prior_dist = MixtureDistribution(
                fsd_xi_prior_log_weights_tuple, fsd_xi_prior_components_tuple)
        else:
            fsd_xi_prior_dist = dist.Normal(fsd_xi_prior_locs[0, :], fsd_xi_prior_scales[0, :]).to_event(1)
        
        with pyro.plate("collapsed_gene_cell", size=mb_size):
            with poutine.scale(scale=data['gene_sampling_site_scale_factor_tensor']):
                # sample gene family size distribution parameters
                fsd_xi = pyro.sample("fsd_xi", fsd_xi_prior_dist)

                # transform to the constrained space
                fsd_params_dict = self.fsd_codec.decode(fsd_xi)

            # get chimeric and real family size distributions
            fsd_lo_dist, fsd_hi_dist = self.fsd_codec.get_fsd_components(
                fsd_params_dict,
                downsampling_rate_tensor=data['downsampling_rate_tensor'])

            # extract required quantities from the distributions
            mu_lo = fsd_lo_dist.mean.squeeze(-1)
            mu_hi = fsd_hi_dist.mean.squeeze(-1)
            log_p_unobs_lo = fsd_lo_dist.log_prob(zero).squeeze(-1)
            log_p_unobs_hi = fsd_hi_dist.log_prob(zero).squeeze(-1)
            log_p_obs_lo = get_log_prob_compl(log_p_unobs_lo)
            log_p_obs_hi = get_log_prob_compl(log_p_unobs_hi)
            p_obs_lo = log_p_obs_lo.exp()
            p_obs_hi = log_p_obs_hi.exp()

            # soft constraints on family size distribution parameters
            
            # localization and/or calculation of required variables for pickup by locals() -- see below
            p_obs_lo_to_p_obs_hi_ratio = p_obs_lo / p_obs_hi
            phi_lo_comps = fsd_params_dict['phi_lo']
            phi_hi_comps = fsd_params_dict['phi_hi']
            mu_lo_comps = fsd_params_dict['mu_lo']
            mu_hi_comps = fsd_params_dict['mu_hi']
            w_lo_comps = fsd_params_dict['w_lo']
            w_hi_comps = fsd_params_dict['w_hi']
            mu_hi_comps_to_mu_empirical_ratio = mu_hi_comps / (
                self.EPS + data['empirical_fsd_mu_hi_tensor'].unsqueeze(-1))
            mu_lo_comps_to_mu_empirical_ratio = mu_lo_comps / (
                self.EPS + data['empirical_fsd_mu_hi_tensor'].unsqueeze(-1))
            
            # observation probability for each component of the distribution
            alpha_lo_comps = (self.EPS + phi_lo_comps).reciprocal()
            log_p_unobs_lo_comps = alpha_lo_comps * (alpha_lo_comps.log() - (alpha_lo_comps + mu_lo_comps).log())
            p_obs_lo_comps = get_log_prob_compl(log_p_unobs_lo_comps).exp()
            alpha_hi_comps = (self.EPS + phi_hi_comps).reciprocal()
            log_p_unobs_hi_comps = alpha_hi_comps * (alpha_hi_comps.log() - (alpha_hi_comps + mu_hi_comps).log())
            p_obs_hi_comps = get_log_prob_compl(log_p_unobs_hi_comps).exp()
            
            # slicing expression mu and phi by gene_index_tensor -- we only need these slices later on
            phi_e_lo_batch = phi_e_lo[data['gene_index_tensor']]
            phi_e_hi_batch = phi_e_hi[data['gene_index_tensor']]
            mu_e_hi_batch = mu_e_hi[data['gene_index_tensor']]
            logit_p_zero_e_hi_batch = logit_p_zero_e_hi[data['gene_index_tensor']]
            
            # total observed molecules
            e_obs = torch.sum(data['fingerprint_tensor'], -1)

            # if running the model in posterior mode or map mode, skip regularizations
            if not (calculate_joint_expression_log_posterior or calculate_expression_map):
                
                # regularizations (on gene-plate quantities)
                with poutine.scale(scale=data['gene_sampling_site_scale_factor_tensor']):
                    
                    # family size distribution sparsity regularization
                    if self.enable_fsd_w_dirichlet_reg:
                        if self.fsd_codec.n_fsd_lo_comps > 1:
                            with poutine.scale(scale=self.w_lo_dirichlet_reg_strength):
                                pyro.sample(
                                    "w_lo_dirichlet_reg",
                                    dist.Dirichlet(self.w_lo_dirichlet_concentration * torch.ones_like(fsd_params_dict['w_lo'])),
                                    obs=fsd_params_dict['w_lo'])
                        if self.fsd_codec.n_fsd_hi_comps > 1:
                            with poutine.scale(scale=self.w_hi_dirichlet_reg_strength):
                                pyro.sample(
                                    "w_hi_dirichlet_reg",
                                    dist.Dirichlet(self.w_hi_dirichlet_concentration * torch.ones_like(fsd_params_dict['w_hi'])),
                                    obs=fsd_params_dict['w_hi'])

                    # (soft) constraints
                    model_vars_dict = locals()
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
                                var_name + "_lower_bound_consraint",
                                CustomLogProbTerm(constraint_log_prob, batch_shape=torch.Size([mb_size]), event_shape=torch.Size([])),
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
                                var_name + "_upper_bound_consraint",
                                CustomLogProbTerm(constraint_log_prob, batch_shape=torch.Size([mb_size]), event_shape=torch.Size([])),
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
                                var_name + "_pin_value_consraint",
                                CustomLogProbTerm(constraint_log_prob, batch_shape=torch.Size([mb_size]), event_shape=torch.Size([])),
                                obs=torch.zeros_like(constraint_log_prob))

            cell_size_scale = data['total_obs_reads_per_cell_tensor'] / (
                self.median_total_reads_per_cell * data['downsampling_rate_tensor'])

            # e_hi prior distribution
            mu_e_hi_per_cell = cell_size_scale * mu_e_hi_batch
            if self.e_hi_prior_dist == 'zinb':
                e_hi_prior_dist_per_cell = ZeroInflatedNegativeBinomial(
                    logit_zero=logit_p_zero_e_hi_batch,
                    mu=mu_e_hi_per_cell,
                    phi=phi_e_hi_batch)
                e_hi_prior_dist_global = ZeroInflatedNegativeBinomial(
                    logit_zero=logit_p_zero_e_hi_batch,
                    mu=mu_e_hi_batch,
                    phi=phi_e_hi_batch)
            elif self.e_hi_prior_dist == 'negbinom':
                e_hi_prior_dist_per_cell = NegativeBinomial(mu=mu_e_hi_per_cell, phi=phi_e_hi_batch)
                e_hi_prior_dist_global = NegativeBinomial(mu=mu_e_hi_batch, phi=phi_e_hi_batch)
            elif self.e_hi_prior_dist == 'poisson':
                e_hi_prior_dist_per_cell = dist.Poisson(mu_e_hi_per_cell)
                e_hi_prior_dist_global = dist.Poisson(mu_e_hi_batch)
            else:
                raise Exception('Bad value for e_hi_prior_dist!')

            # calculating mean chimeric counts per cell
            normalized_total_fragments = e_hi_prior_dist_global.mean * mu_hi / (
                self.median_fsd_mu_hi * data['downsampling_rate_tensor'])
            mu_e_lo = (alpha_c + beta_c * cell_size_scale) * normalized_total_fragments
            
            # e_lo prior distribution
            if self.e_lo_prior_dist == 'negbinom':
                e_lo_prior_dist_per_cell = NegativeBinomial(mu=mu_e_lo, phi=phi_e_lo_batch)
            elif self.e_lo_prior_dist == 'poisson':
                e_lo_prior_dist_per_cell = dist.Poisson(mu_e_lo)
            else:
                raise Exception('Bad value for e_lo_prior_dist!')

            # e_lo and e_hi marginalization range
            if 'e_lo_min' in data.keys() and 'e_hi_min' in data.keys():
                e_lo_min = data['e_lo_min']
                e_hi_min = data['e_hi_min']
            else:
                # estimate chimera and real expression ranges
                with torch.no_grad():
                    e_lo_min = torch.ceil(torch.clamp(mu_e_lo - 0.5 * e_lo_sum_width, 0.))
                    e_hi_min = torch.ceil(torch.clamp(
                        (e_obs - p_obs_lo * mu_e_lo) / p_obs_hi - 0.5 * e_hi_sum_width, 0.))

            e_lo = e_lo_min + e_lo_range.view(-1, 1)
            e_hi = e_hi_min + e_hi_range.view(-1, 1, 1)
            
            # e_lo and e_hi prior log probs
            e_lo_log_prob = e_lo_prior_dist_per_cell.log_prob(e_lo)
            e_hi_log_prob = e_hi_prior_dist_per_cell.log_prob(e_hi)
            
            if self.model_type == 'approx_multinomial':
                # mixture weights
                log_w_lo = (e_lo / (e_lo + e_hi + self.EPS)).log()
                log_w_hi = get_log_prob_compl(log_w_lo)

                # calculate the fit log prob
                dist_mixture = MixtureDistribution(
                    (log_w_lo.unsqueeze(-1), log_w_hi.unsqueeze(-1)),
                    (fsd_lo_dist, fsd_hi_dist), normalize_weights=False)
                fit_log_prob = dist_mixture.log_prob(family_size_vector_observable)

                # observing the fingerprint (multinomial)
                normalized_fit_log_prob = fit_log_prob - torch.logsumexp(fit_log_prob, -1, keepdim=True)
                log_prob_fingerprint_obs = (
                    (e_obs + 1).lgamma()
                    - (data['fingerprint_tensor'] + 1).lgamma().sum(-1)
                    + (data['fingerprint_tensor'] * normalized_fit_log_prob).sum(-1))

                # observing total expression (binomial)
                log_p_unobs = dist_mixture.log_prob(zero).squeeze(-1)
                log_p_obs = get_log_prob_compl(log_p_unobs)
                e_total = e_lo + e_hi
                log_prob_e_obs = (
                    (e_total + 1).lgamma() - (e_obs + 1).lgamma() - (e_total - e_obs + 1).lgamma()
                    + e_obs * log_p_obs + (e_total - e_obs) * log_p_unobs)            

                # total observation log prob
                log_prob_total_obs = (
                    data['e_lo_log_prob_prefactor'] * e_lo_log_prob +
                    data['e_hi_log_prob_prefactor'] * e_hi_log_prob +
                    data['e_obs_log_prob_prefactor'] * log_prob_e_obs +
                    data['fingerprint_obs_log_prob_prefactor'] * log_prob_fingerprint_obs)
                
            # todo: get rid of EPS properly ...
            elif self.model_type == 'poisson':
                log_prob_p_lo_full = fsd_lo_dist.log_prob(family_size_vector_full)
                log_prob_p_hi_full = fsd_hi_dist.log_prob(family_size_vector_full)
                
                fingerprint_log_rate_full = logaddexp(
                    (e_lo.unsqueeze(-1) + self.EPS).log() + log_prob_p_lo_full,
                    (e_hi.unsqueeze(-1) + self.EPS).log() + log_prob_p_hi_full)
                fingerprint_log_rate_obs = fingerprint_log_rate_full[..., 1:]
                fingerprint_log_rate_unobs = fingerprint_log_rate_full[..., 0]
                
                # observing the fingerprint (poisson)
                log_prob_fingerprint_obs = (
                    (data['fingerprint_tensor'] * fingerprint_log_rate_obs).sum(-1)
                    - fingerprint_log_rate_obs.exp().sum(-1))
#                     - (data['fingerprint_tensor'] + 1).lgamma().sum(-1))

                # total observation log prob
                log_prob_total_obs = (
                    data['e_lo_log_prob_prefactor'] * e_lo_log_prob +
                    data['e_hi_log_prob_prefactor'] * e_hi_log_prob +
                    # data['e_obs_log_prob_prefactor'] * log_prob_e_obs +
                    data['fingerprint_obs_log_prob_prefactor'] * log_prob_fingerprint_obs)

                # normalized fit log prob (auxiliary quantity)
                normalized_fit_log_prob = (
                    fingerprint_log_rate_obs
                    - torch.logsumexp(fingerprint_log_rate_obs, -1, keepdim=True))

            else:
                raise ValueError(f'Unknown model type "{self.model_type}"!')

            # kill unphysical paths (... which may have been made available due to approximations)
            log_prob_total_obs[(e_lo + e_hi) < e_obs] = - float("inf")

            # marginalize
            marginalized_log_prob_total_obs = torch.logsumexp(torch.logsumexp(log_prob_total_obs, 0), 0)

            # expression MAP inference
            if calculate_expression_map:
                with torch.no_grad():    
                    # calculating MAP
                    e_lo_bc, e_hi_bc, _ = broadcast_all(e_lo, e_hi, log_prob_total_obs)
                    e_lo_flat = e_lo_bc.long().contiguous().view(-1, mb_size)
                    e_hi_flat = e_hi_bc.long().contiguous().view(-1, mb_size)
                    argmax_res = torch.argmax(log_prob_total_obs.contiguous().view(-1, mb_size), dim=0)
                    e_lo_map = e_lo_flat[argmax_res, np.arange(mb_size)]
                    e_hi_map = e_hi_flat[argmax_res, np.arange(mb_size)]
                    fit_log_prob_map = normalized_fit_log_prob.view(-1, mb_size, max_family_size)[argmax_res, np.arange(mb_size), :]

                    # calculating posterior confidence intervals
                    log_prob_normalized_posterior = log_prob_total_obs - marginalized_log_prob_total_obs
                    
                    e_lo_log_posterior = torch.logsumexp(log_prob_normalized_posterior, 0)
                    e_lo_log_posterior = e_lo_log_posterior - torch.logsumexp(e_lo_log_posterior, 0)
                    e_lo_posterior_pdf = e_lo_log_posterior.exp()
                    e_lo_posterior_cdf = torch.cumsum(e_lo_posterior_pdf, 0)

                    e_hi_log_posterior = torch.logsumexp(log_prob_normalized_posterior, 1)
                    e_hi_log_posterior = e_hi_log_posterior - torch.logsumexp(e_hi_log_posterior, 0)
                    e_hi_posterior_pdf = e_hi_log_posterior.exp()
                    e_hi_posterior_cdf = torch.cumsum(e_hi_posterior_pdf, 0)
                    
                    e_lo_ci_lower, e_lo_ci_upper = get_confidence_interval(
                        e_lo_posterior_cdf, self.DEFAULT_CONFIDENCE_INTERVAL_LOWER, self.DEFAULT_CONFIDENCE_INTERVAL_UPPER)
                    e_lo_ci_lower += e_lo_min.long()
                    e_lo_ci_upper += e_lo_min.long()
                    
                    e_hi_ci_lower, e_hi_ci_upper = get_confidence_interval(
                        e_hi_posterior_cdf, self.DEFAULT_CONFIDENCE_INTERVAL_LOWER, self.DEFAULT_CONFIDENCE_INTERVAL_UPPER)
                    e_hi_ci_lower += e_hi_min.long()
                    e_hi_ci_upper += e_hi_min.long()
                    
                    # calculating posterior mean and variance
                    e_lo_mean = (e_lo_posterior_pdf * e_lo_range.unsqueeze(-1)).sum(0)
                    e_lo_var = (e_lo_posterior_pdf * e_lo_range.unsqueeze(-1).pow(2)).sum(0) - e_lo_mean.pow(2)
                    e_lo_mean += e_lo_min
                    
                    e_hi_mean = (e_hi_posterior_pdf * e_hi_range.unsqueeze(-1)).sum(0)
                    e_hi_var = (e_hi_posterior_pdf * e_hi_range.unsqueeze(-1).pow(2)).sum(0) - e_hi_mean.pow(2)
                    e_hi_mean += e_hi_min
                    
                    return {'e_lo_min': e_lo_min.long(),
                            'e_hi_min': e_hi_min.long(),
                            'e_lo_map': e_lo_map.long(),
                            'e_hi_map': e_hi_map.long(),
                            'e_lo_ci_lower': e_lo_ci_lower,
                            'e_lo_ci_upper': e_lo_ci_upper,
                            'e_hi_ci_lower': e_hi_ci_lower,
                            'e_hi_ci_upper': e_hi_ci_upper,
                            'e_lo_mean': e_lo_mean,
                            'e_lo_var': e_lo_var,
                            'e_hi_mean': e_hi_mean,
                            'e_hi_var': e_hi_var,                            
                            'fit_log_prob_map': fit_log_prob_map,
                            'mu_e_lo': mu_e_lo}
            
            # expression PM inference
            if calculate_joint_expression_log_posterior:
                # normalize the posterior
                log_prob_normalized_posterior = log_prob_total_obs - marginalized_log_prob_total_obs
                return {'e_lo_min': e_lo_min,
                        'e_hi_min': e_hi_min,
                        'log_prob_normalized_posterior': log_prob_normalized_posterior}

            # observe
            with poutine.scale(scale=data['cell_sampling_site_scale_factor_tensor']):
                pyro.sample("fingerprint_and_expression_observation",
                            CustomLogProbTerm(
                                custom_log_prob=marginalized_log_prob_total_obs,
                                batch_shape=marginalized_log_prob_total_obs.shape,
                                event_shape=torch.Size([])),
                            obs=torch.zeros_like(marginalized_log_prob_total_obs))
                            
    def guide(self,
              data,
              calculate_expression_map=False,
              calculate_joint_expression_log_posterior=False):
        if self.fsd_gmm_num_components > 1:
            # MAP estimate of GMM fsd prior weights
            fsd_xi_prior_weights_map = pyro.param(
                "fsd_xi_prior_weights_map",
                torch.ones((self.fsd_gmm_num_components,),
                           device=self.device, dtype=self.dtype) / self.fsd_gmm_num_components,
                constraint=constraints.simplex)
            fsd_xi_prior_weights = pyro.sample("fsd_xi_prior_weights", dist.Delta(
                self.fsd_gmm_min_weight_per_component +
                (1 - self.fsd_gmm_num_components * self.fsd_gmm_min_weight_per_component) * fsd_xi_prior_weights_map))

        # point estimate for fsd_xi (gene)
        fsd_xi_posterior_loc = pyro.param(
            "fsd_xi_posterior_loc",
            self.fsd_codec.get_sorted_fsd_xi(self.fsd_codec.init_fsd_xi_loc_posterior))
        
        # base posterior distribution for xi
        if self.guide_type == 'map' or calculate_expression_map:
            fsd_xi_posterior_base_dist = dist.Delta(
                v=fsd_xi_posterior_loc[data['gene_index_tensor'], :]).to_event(1)
        elif self.guide_type == 'gaussian':
            fsd_xi_posterior_scale = pyro.param(
                "fsd_xi_posterior_scale",
                self.fsd_gmm_init_xi_scale * torch.ones(
                    (self.n_total_genes, self.fsd_codec.total_fsd_params), device=self.device, dtype=self.dtype),
                constraint=constraints.greater_than(self.fsd_xi_posterior_min_scale))
            fsd_xi_posterior_base_dist = dist.Normal(
                loc=fsd_xi_posterior_loc[data['gene_index_tensor'], :],
                scale=fsd_xi_posterior_scale[data['gene_index_tensor'], :]).to_event(1)
        else:
            raise Exception("Unknown guide_type!")
        
        # apply a pseudo-bijective transformation to sort xi by component weights
        fsd_xi_sort_trans = SortByComponentWeights(self.fsd_codec)
        fsd_xi_posterior_dist = dist.TransformedDistribution(
            fsd_xi_posterior_base_dist, [fsd_xi_sort_trans])
        
        mb_size = data['fingerprint_tensor'].shape[0]
        with pyro.plate("collapsed_gene_cell", size=mb_size):
            with poutine.scale(scale=data['gene_sampling_site_scale_factor_tensor']):
                fsd_xi = pyro.sample("fsd_xi", fsd_xi_posterior_dist)

    def get_active_constraints_on_genes(self) -> Dict:
        empirical_fsd_mu_hi_tensor = torch.tensor(
            self.sc_fingerprint_datastore.empirical_fsd_mu_hi, device=self.device, dtype=self.dtype)
        zero = torch.tensor(0, device=self.device, dtype=self.dtype)
        active_constraints_dict = defaultdict(dict)

        with torch.no_grad():
            fsd_xi = pyro.param("fsd_xi_posterior_loc")

            # transform to the constrained space
            fsd_params_dict = self.fsd_codec.decode(fsd_xi)

            # get chimeric and real family size distributions
            fsd_lo_dist, fsd_hi_dist = self.fsd_codec.get_fsd_components(fsd_params_dict, None)

            # extract required quantities from the distributions
            mu_lo = fsd_lo_dist.mean.squeeze(-1)
            mu_hi = fsd_hi_dist.mean.squeeze(-1)
            log_p_unobs_lo = fsd_lo_dist.log_prob(zero).squeeze(-1)
            log_p_unobs_hi = fsd_hi_dist.log_prob(zero).squeeze(-1)
            log_p_obs_lo = get_log_prob_compl(log_p_unobs_lo)
            log_p_obs_hi = get_log_prob_compl(log_p_unobs_hi)
            p_obs_lo = log_p_obs_lo.exp()
            p_obs_hi = log_p_obs_hi.exp()

            # localization and/or calculation of required variables for pickup by locals()
            p_obs_lo_to_p_obs_hi_ratio = p_obs_lo / p_obs_hi
            phi_lo_comps = fsd_params_dict['phi_lo']
            phi_hi_comps = fsd_params_dict['phi_hi']
            mu_lo_comps = fsd_params_dict['mu_lo']
            mu_hi_comps = fsd_params_dict['mu_hi']
            w_lo_comps = fsd_params_dict['w_lo']
            w_hi_comps = fsd_params_dict['w_hi']
            mu_hi_comps_to_mu_empirical_ratio = mu_hi_comps / (
                self.EPS + empirical_fsd_mu_hi_tensor.unsqueeze(-1))
            mu_lo_comps_to_mu_empirical_ratio = mu_lo_comps / (
                self.EPS + empirical_fsd_mu_hi_tensor.unsqueeze(-1))
            alpha_lo_comps = (self.EPS + phi_lo_comps).reciprocal()
            log_p_unobs_lo_comps = alpha_lo_comps * (alpha_lo_comps.log() - (alpha_lo_comps + mu_lo_comps).log())
            p_obs_lo_comps = get_log_prob_compl(log_p_unobs_lo_comps).exp()
            alpha_hi_comps = (self.EPS + phi_hi_comps).reciprocal()
            log_p_unobs_hi_comps = alpha_hi_comps * (alpha_hi_comps.log() - (alpha_hi_comps + mu_hi_comps).log())
            p_obs_hi_comps = get_log_prob_compl(log_p_unobs_hi_comps).exp()
            phi_e_lo_batch = pyro.param("phi_e_lo")
            phi_e_hi_batch = pyro.param("phi_e_hi")
            mu_e_hi_batch = pyro.param("mu_e_hi")
            logit_p_zero_e_hi_batch = pyro.param("logit_p_zero_e_hi")

            model_vars_dict = locals()
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
                    exponent = var_constraint_params['upper_bound_exponent']
                    strength = var_constraint_params['upper_bound_strength']
                    if isinstance(value, str):
                        value = model_vars_dict[value]
                    activity = torch.clamp(var - value + width, min=0.)
                    for _ in range(len(var.shape) - 1):
                        activity = activity.sum(-1)
                    nnz_activity = torch.nonzero(activity).cpu().numpy().flatten()
                    if nnz_activity.size > 0:
                        active_constraints_dict[var_name]['upper_bound'] = set(nnz_activity.tolist())

        return dict(active_constraints_dict)
        

class DownsamplingRegularizedELBOLoss:
    def __init__(self,
                 downsampling_regularization_strength: float,
                 min_downsampling_rate: float,
                 max_downsampling_rate: float,
                 keep_history=False,
                 disable_downsampling_regularization=False):
        self.downsampling_regularization_strength = downsampling_regularization_strength
        self.min_downsampling_rate = min_downsampling_rate
        self.max_downsampling_rate = max_downsampling_rate
        self.elbo = Trace_ELBO()
        self.keep_history = keep_history
        self.disable_downsampling_regularization = disable_downsampling_regularization
        
        self.elbo_loss_history = []
        self.downsampling_loss_history = []
        
    def differentiable_loss(self,
                            model,
                            guide,
                            mb_data_elbo: Dict[str, torch.Tensor],
                            mb_data_reg: Dict[str, torch.Tensor]):
        # - ELBO calculated on mb_data_elbo
        elbo_loss = self.elbo.differentiable_loss(model, guide, mb_data_elbo)
        total_loss = elbo_loss
        
        if self.keep_history:
            self.elbo_loss_history.append(elbo_loss.item())
            
        # downsampling regularization calculated on mb_data_reg
        if not self.disable_downsampling_regularization:
            # step 1. obtain the joint posterior (e_lo, e_hi) from mb_data_reg
            guide_trace_on_mb_data_reg = poutine.trace(guide).get_trace(mb_data_reg)
            trained_model_on_mb_data_reg = poutine.replay(model, trace=guide_trace_on_mb_data_reg)
            inference_on_mb_data_reg = poutine.trace(trained_model_on_mb_data_reg).get_trace(
                mb_data_reg, calculate_joint_expression_log_posterior=True).nodes["_RETURN"]["value"]

            # step 2. downsample mb_data_reg
            ds_mb_data_reg = generate_downsampled_minibatch(
                mb_data_reg,
                self.min_downsampling_rate,
                self.max_downsampling_rate)

            # step 3. insert e_lo_min and e_hi_min on ds_mb_data_reg so that both the
            # model uses the same marginalization bounds on the original and downsampled minibatch
            ds_mb_data_reg['e_lo_min'] = inference_on_mb_data_reg['e_lo_min']
            ds_mb_data_reg['e_hi_min'] = inference_on_mb_data_reg['e_hi_min']

            # step 4. obtain the joint posterior (e_lo, e_hi) from ds_mb_data_reg
            guide_trace_on_ds_mb_data_reg = poutine.trace(guide).get_trace(ds_mb_data_reg)
            trained_model_on_ds_mb_data_reg = poutine.replay(model, trace=guide_trace_on_ds_mb_data_reg)
            inference_on_ds_mb_data_reg = poutine.trace(trained_model_on_ds_mb_data_reg).get_trace(
                ds_mb_data_reg, calculate_joint_expression_log_posterior=True).nodes["_RETURN"]["value"]

            # step 5. calculate the distance between original and downsampled posteriors
            joint_expression_posterior_ds_divergence = get_hellinger_distance(
                inference_on_mb_data_reg['log_prob_normalized_posterior'],
                inference_on_ds_mb_data_reg['log_prob_normalized_posterior'],
                reduce=lambda x: x.sum(0).sum(0))
            downsampling_loss = self.downsampling_regularization_strength * (
                mb_data_reg['cell_sampling_site_scale_factor_tensor'] *
                joint_expression_posterior_ds_divergence).sum()
        
            total_loss += downsampling_loss
            
            if self.keep_history:
                self.downsampling_loss_history.append(downsampling_loss.item())
        
        return total_loss
    
    def reset_history(self):
        self.elbo_loss_history = []
        self.downsampling_loss_history = []

        
def get_expression_map(gene_index,
                       model,
                       sc_fingerprint_datastore,
                       e_lo_sum_width,
                       e_hi_sum_width,
                       cell_shard_size,
                       device=torch.device('cuda'),
                       dtype=torch.float,
                       **model_args_override):
    e_lo_min_shard_list = []
    e_hi_min_shard_list = []
    e_lo_map_shard_list = []
    e_hi_map_shard_list = []
    e_lo_ci_lower_shard_list = []
    e_hi_ci_lower_shard_list = []
    e_lo_ci_upper_shard_list = []
    e_hi_ci_upper_shard_list = []
    e_lo_mean_shard_list = []
    e_hi_mean_shard_list = []
    e_lo_var_shard_list = []
    e_hi_var_shard_list = []
    fit_log_prob_map_shard_list = []
    mu_e_lo_shard_list = []
    
    def _fix_scalar(arr):
        if arr.ndim == 0:
            arr = arr[None]
        return arr
    
    n_cells = sc_fingerprint_datastore.n_cells
    for cell_shard in range(n_cells // cell_shard_size + 1):
        i_cell_begin = min(cell_shard * cell_shard_size, n_cells)
        i_cell_end = min((cell_shard + 1) * cell_shard_size, n_cells)
        if i_cell_begin == i_cell_end:
            break

        # generate shard data
        cell_index_array = np.arange(i_cell_begin, i_cell_end)
        gene_index_array = gene_index * np.ones_like(cell_index_array)
        cell_sampling_site_scale_factor_array = np.ones_like(cell_index_array)
        gene_sampling_site_scale_factor_array = np.ones_like(cell_index_array)
        shard_data = sc_fingerprint_datastore.generate_torch_minibatch_data(
            cell_index_array,
            gene_index_array,
            cell_sampling_site_scale_factor_array,
            gene_sampling_site_scale_factor_array,
            device,
            dtype)
        shard_data['e_lo_sum_width'] = e_lo_sum_width
        shard_data['e_hi_sum_width'] = e_hi_sum_width
        
        for k, v in model_args_override.items():
            shard_data[k] = v

        guide_trace = poutine.trace(model.guide).get_trace(shard_data, calculate_expression_map=True)
        trained_model = poutine.replay(model.model, trace=guide_trace)
        trace = poutine.trace(trained_model).get_trace(shard_data, calculate_expression_map=True)

        # download MAP estimates to numpy
        e_lo_map = trace.nodes["_RETURN"]["value"]["e_lo_map"].cpu().numpy().squeeze()
        e_hi_map = trace.nodes["_RETURN"]["value"]["e_hi_map"].cpu().numpy().squeeze()
        e_lo_min = trace.nodes["_RETURN"]["value"]["e_lo_min"].cpu().numpy().squeeze()
        e_hi_min = trace.nodes["_RETURN"]["value"]["e_hi_min"].cpu().numpy().squeeze()
        e_lo_ci_lower = trace.nodes["_RETURN"]["value"]["e_lo_ci_lower"].cpu().numpy().squeeze()
        e_hi_ci_lower = trace.nodes["_RETURN"]["value"]["e_hi_ci_lower"].cpu().numpy().squeeze()
        e_lo_ci_upper = trace.nodes["_RETURN"]["value"]["e_lo_ci_upper"].cpu().numpy().squeeze()
        e_hi_ci_upper = trace.nodes["_RETURN"]["value"]["e_hi_ci_upper"].cpu().numpy().squeeze()
        e_lo_mean = trace.nodes["_RETURN"]["value"]["e_lo_mean"].cpu().numpy().squeeze()
        e_hi_mean = trace.nodes["_RETURN"]["value"]["e_hi_mean"].cpu().numpy().squeeze()
        e_lo_var = trace.nodes["_RETURN"]["value"]["e_lo_var"].cpu().numpy().squeeze()
        e_hi_var = trace.nodes["_RETURN"]["value"]["e_hi_var"].cpu().numpy().squeeze()
        fit_log_prob_map = trace.nodes["_RETURN"]["value"]["fit_log_prob_map"].detach().cpu().numpy()
        mu_e_lo = trace.nodes["_RETURN"]["value"]["mu_e_lo"].detach().cpu().numpy()
        
        e_lo_map_shard_list.append(_fix_scalar(e_lo_map))
        e_hi_map_shard_list.append(_fix_scalar(e_hi_map))
        e_lo_min_shard_list.append(_fix_scalar(e_lo_min))
        e_hi_min_shard_list.append(_fix_scalar(e_hi_min))
        e_lo_ci_lower_shard_list.append(_fix_scalar(e_lo_ci_lower))
        e_hi_ci_lower_shard_list.append(_fix_scalar(e_hi_ci_lower))
        e_lo_ci_upper_shard_list.append(_fix_scalar(e_lo_ci_upper))
        e_hi_ci_upper_shard_list.append(_fix_scalar(e_hi_ci_upper))
        e_lo_mean_shard_list.append(_fix_scalar(e_lo_mean))
        e_hi_mean_shard_list.append(_fix_scalar(e_hi_mean))
        e_lo_var_shard_list.append(_fix_scalar(e_lo_var))
        e_hi_var_shard_list.append(_fix_scalar(e_hi_var))
        fit_log_prob_map_shard_list.append(_fix_scalar(fit_log_prob_map))
        mu_e_lo_shard_list.append(_fix_scalar(mu_e_lo))
                    
    return {'e_lo_map': np.concatenate(e_lo_map_shard_list),
            'e_hi_map': np.concatenate(e_hi_map_shard_list),
            'e_lo_min': np.concatenate(e_lo_min_shard_list),
            'e_hi_min': np.concatenate(e_hi_min_shard_list),
            'e_lo_ci_lower': np.concatenate(e_lo_ci_lower_shard_list),
            'e_hi_ci_lower': np.concatenate(e_hi_ci_lower_shard_list),
            'e_lo_ci_upper': np.concatenate(e_lo_ci_upper_shard_list),
            'e_hi_ci_upper': np.concatenate(e_hi_ci_upper_shard_list),
            'e_lo_mean': np.concatenate(e_lo_mean_shard_list),
            'e_hi_mean': np.concatenate(e_hi_mean_shard_list),
            'e_lo_var': np.concatenate(e_lo_var_shard_list),
            'e_hi_var': np.concatenate(e_hi_var_shard_list),
            'fit_log_prob_map': np.concatenate(fit_log_prob_map_shard_list),
            'mu_e_lo': np.concatenate(mu_e_lo_shard_list)}
