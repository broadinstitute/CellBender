import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator, Optional
from abc import abstractmethod

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
from pyro.nn.module import PyroParam, pyro_method
from pyro.contrib.gp.parameterized import Parameterized

from fingerprint import SingleCellFingerprintDTM
from stats import gamma_loc_scale_to_concentration_rate


class ChimeraRateModel(Parameterized):
    def __init__(self):
        super(ChimeraRateModel, self).__init__()

    @abstractmethod
    def model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def decode_output_to_chimera_rate(
            self,
            output_dict: Dict[str, torch.Tensor],
            data_dict: Dict[str, torch.Tensor],
            parents_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class UniformChimeraRateModel(ChimeraRateModel):
    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 init_params_dict: Dict[str, float],
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float32):
        super(UniformChimeraRateModel, self).__init__()

        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.device = device
        self.dtype = dtype

        self.alpha_c_prior_a, self.alpha_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['chimera.alpha_c_prior_loc'],
            scale=init_params_dict['chimera.alpha_c_prior_scale'])

        self.beta_c_prior_a, self.beta_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['chimera.beta_c_prior_loc'],
            scale=init_params_dict['chimera.beta_c_prior_scale'])

        self.detach_non_hvg_genes = init_params_dict['chimera.detach_non_hvg_genes']

        # chimera hyperparameters
        self.alpha_c_concentration_scalar = torch.tensor(
            self.alpha_c_prior_a, device=self.device, dtype=self.dtype)
        self.alpha_c_rate_scalar = torch.tensor(
            self.alpha_c_prior_b, device=self.device, dtype=self.dtype)
        self.beta_c_concentration_scalar = torch.tensor(
            self.beta_c_prior_a, device=self.device, dtype=self.dtype)
        self.beta_c_rate_scalar = torch.tensor(
            self.beta_c_prior_b, device=self.device, dtype=self.dtype)

        # empirical normalization factors
        self.mean_empirical_fsd_mu_hi: float = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi).item()

        # trainable parameters
        self.alpha_c_posterior_loc = PyroParam(
            torch.tensor(self.alpha_c_prior_a / self.alpha_c_prior_b, device=self.device, dtype=self.dtype),
            constraints.positive)

        self.beta_c_posterior_loc = PyroParam(
            torch.tensor(self.beta_c_prior_a / self.beta_c_prior_b, device=self.device, dtype=self.dtype),
            constraints.positive)

    @pyro_method
    @autoname.scope(prefix="uniform_chimera")
    def model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.set_mode('model')

        # sample chimera rate parameters
        alpha_c = pyro.sample(
            "alpha_c",
            dist.Gamma(
                concentration=self.alpha_c_concentration_scalar,
                rate=self.alpha_c_rate_scalar))
        beta_c = pyro.sample(
            "beta_c",
            dist.Gamma(
                concentration=self.beta_c_concentration_scalar,
                rate=self.beta_c_rate_scalar))

        return {
            'alpha_c': alpha_c,
            'beta_c': beta_c
        }

    @pyro_method
    @autoname.scope(prefix="uniform_chimera")
    def guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.set_mode('guide')

        alpha_c = pyro.sample("alpha_c", dist.Delta(v=self.alpha_c_posterior_loc))
        beta_c = pyro.sample("beta_c", dist.Delta(v=self.beta_c_posterior_loc))

        return {
            'alpha_c': alpha_c,
            'beta_c': beta_c
        }

    @abstractmethod
    def decode_output_to_chimera_rate(
            self,
            output_dict: Dict[str, torch.Tensor],
            data_dict: Dict[str, torch.Tensor],
            parents_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        assert 'alpha_c' in output_dict
        assert 'beta_c' in output_dict

        assert 'mu_fsd_hi_n' in parents_dict
        assert 'eta_n' in parents_dict
        assert 'mu_e_hi_cell_averaged_n' in parents_dict

        alpha_c = output_dict['alpha_c']
        beta_c = output_dict['beta_c']

        if self.detach_non_hvg_genes:

            assert 'hvg_binary_mask_tensor_n' in parents_dict
            hvg_binary_mask_tensor_n = parents_dict['hvg_binary_mask_tensor_n'].float()
            non_hvg_binary_mask_tensor_n = (~parents_dict['hvg_binary_mask_tensor_n']).float()

            alpha_c_detached = alpha_c.clone().detach()
            beta_c_detached = beta_c.clone().detach()

            alpha_c_n = (
                alpha_c * hvg_binary_mask_tensor_n.float()
                + alpha_c_detached * non_hvg_binary_mask_tensor_n)
            beta_c_n = (
                beta_c * hvg_binary_mask_tensor_n.float()
                + beta_c_detached * non_hvg_binary_mask_tensor_n)

        else:

            alpha_c_n = alpha_c
            beta_c_n = beta_c

        mu_fsd_hi_n = parents_dict['mu_fsd_hi_n']
        eta_n = parents_dict['eta_n']
        mu_e_hi_cell_averaged_n = parents_dict['mu_e_hi_cell_averaged_n']

        # calculate chimera rate
        scaled_mu_fsd_hi_n = mu_fsd_hi_n
        rho_n = (alpha_c_n + beta_c_n * eta_n) * scaled_mu_fsd_hi_n
        mu_e_lo_n = rho_n * mu_e_hi_cell_averaged_n

        return {
            'mu_e_lo_n': mu_e_lo_n
        }

# class GeneLevelChimeraRateModel(ChimeraRateModel):
#     def __init__(self,
#                  sc_fingerprint_dtm: SingleCellFingerprintDTM,
#                  init_params_dict: Dict[str, float],
#                  trainable_prior: bool = True,
#                  device: torch.device = torch.device("cuda"),
#                  dtype: torch.dtype = torch.float32):
#         super(GeneLevelChimeraRateModel, self).__init__()
#
#         self.sc_fingerprint_dtm = sc_fingerprint_dtm
#         self.trainable_prior = trainable_prior
#         self.device = device
#         self.dtype = dtype
#
#         if trainable_prior:
#
#             self.log_alpha_c_prior_loc = PyroParam(
#                 torch.tensor(
#                     np.log(init_params_dict['chimera.alpha_c_prior_loc']),
#                     device=device, dtype=dtype))
#             self.log_alpha_c_prior_scale = PyroParam(
#                 torch.tensor(
#                     init_params_dict['chimera.alpha_c_prior_scale'],
#                     device=device, dtype=dtype),
#                 constraints.positive)
#
#             self.log_beta_c_prior_loc = PyroParam(
#                 torch.tensor(
#                     np.log(init_params_dict['chimera.beta_c_prior_loc']),
#                     device=device, dtype=dtype))
#             self.log_beta_c_prior_scale = PyroParam(
#                 torch.tensor(
#                     init_params_dict['chimera.beta_c_prior_scale'],
#                     device=device, dtype=dtype),
#                 constraints.positive)
#
#         else:
#
#             self.log_alpha_c_prior_loc = torch.tensor(
#                 np.log(init_params_dict['chimera.alpha_c_prior_loc']),
#                 device=device, dtype=dtype)
#             self.log_alpha_c_prior_scale = torch.tensor(
#                 init_params_dict['chimera.alpha_c_prior_scale'],
#                 device=device, dtype=dtype)
#
#             self.log_beta_c_prior_loc = torch.tensor(
#                 np.log(init_params_dict['chimera.beta_c_prior_loc']),
#                 device=device, dtype=dtype)
#             self.log_beta_c_prior_scale = torch.tensor(
#                 init_params_dict['chimera.beta_c_prior_scale'],
#                 device=device, dtype=dtype)
#
#         # empirical normalization factors
#         self.mean_empirical_fsd_mu_hi: float = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi).item()
#
#         # trainable parameters
#         self.log_alpha_c_posterior_loc_g = PyroParam(
#             np.log(init_params_dict['chimera.alpha_c_prior_loc']) * torch.ones(
#                 sc_fingerprint_dtm.n_genes, device=device, dtype=dtype))
#         self.log_beta_c_posterior_loc_g = PyroParam(
#             np.log(init_params_dict['chimera.beta_c_prior_loc']) * torch.ones(
#                 sc_fingerprint_dtm.n_genes, device=device, dtype=dtype))
#
#     @pyro_method
#     @autoname.scope(prefix="gene_level_chimera")
#     def model(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         self.set_mode('model')
#
#         assert 'gene_index_tensor' in data
#         assert 'gene_sampling_site_scale_factor_tensor' in data
#
#         gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
#         gene_index_tensor_n = data['gene_index_tensor']
#         mb_size = gene_index_tensor_n.shape[0]
#
#         # sample chimera rate parameters
#         with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
#             log_alpha_c_n = pyro.sample(
#                 "log_alpha_c_n",
#                 dist.Normal(
#                     loc=self.log_alpha_c_prior_loc.expand([mb_size]),
#                     scale=self.log_alpha_c_prior_scale.expand([mb_size])))
#             log_beta_c_n = pyro.sample(
#                 "log_beta_c_n",
#                 dist.Normal(
#                     loc=self.log_beta_c_prior_loc.expand([mb_size]),
#                     scale=self.log_beta_c_prior_scale.expand([mb_size])))
#
#         return {
#             'log_alpha_c_n': log_alpha_c_n,
#             'log_beta_c_n': log_beta_c_n
#         }
#
#     @pyro_method
#     @autoname.scope(prefix="gene_level_chimera")
#     def guide(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         self.set_mode('guide')
#
#         assert 'gene_index_tensor' in data
#         assert 'gene_sampling_site_scale_factor_tensor' in data
#
#         gene_sampling_site_scale_factor_tensor_n = data['gene_sampling_site_scale_factor_tensor']
#         gene_index_tensor_n = data['gene_index_tensor']
#
#         # sample chimera rate parameters
#         with poutine.scale(scale=gene_sampling_site_scale_factor_tensor_n):
#             log_alpha_c_n = pyro.sample(
#                 "log_alpha_c_n",
#                 dist.Delta(v=self.log_alpha_c_posterior_loc_g[gene_index_tensor_n]))
#             log_beta_c_n = pyro.sample(
#                 "log_beta_c_n",
#                 dist.Delta(v=self.log_beta_c_posterior_loc_g[gene_index_tensor_n]))
#
#         return {
#             'log_alpha_c_n': log_alpha_c_n,
#             'log_beta_c_n': log_beta_c_n
#         }
#
#     # TODO: needs to be updated to look like the implementation in UniformChimeraRateModel
#     @abstractmethod
#     def decode_output_to_chimera_rate(
#             self,
#             output_dict: Dict[str, torch.Tensor],
#             data_dict: Dict[str, torch.Tensor],
#             parents_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#
#         raise NotImplementedError
#
#         assert 'log_alpha_c_n' in output_dict
#         assert 'log_beta_c_n' in output_dict
#
#         assert 'mu_fsd_hi_n' in parents_dict
#         assert 'eta_n' in parents_dict
#         assert 'total_obs_gene_expr_per_cell_n' in parents_dict
#         assert 'p_obs_lo_n' in parents_dict
#         assert 'p_obs_hi_n' in parents_dict
#
#         log_alpha_c_n = output_dict['log_alpha_c_n']
#         log_beta_c_n = output_dict['log_beta_c_n']
#
#         mu_fsd_hi_n = parents_dict['mu_fsd_hi_n']
#         eta_n = parents_dict['eta_n']
#         total_obs_gene_expr_per_cell_n = parents_dict['total_obs_gene_expr_per_cell_n']
#         p_obs_lo_n = parents_dict['p_obs_lo_n']
#         p_obs_hi_n = parents_dict['p_obs_hi_n']
#
#         # calculate chimera rate
#         alpha_c_n = log_alpha_c_n.exp()
#         beta_c_n = log_beta_c_n.exp()
#         scaled_mu_fsd_hi_n = mu_fsd_hi_n / self.mean_empirical_fsd_mu_hi
#         rho_n = (alpha_c_n + beta_c_n * eta_n) * scaled_mu_fsd_hi_n
#         rho_ave_n = (alpha_c_n + beta_c_n) * scaled_mu_fsd_hi_n
#         total_fragments_n = total_obs_gene_expr_per_cell_n / (rho_ave_n * p_obs_lo_n + p_obs_hi_n)
#         mu_e_lo_n = rho_n * total_fragments_n
#
#         # prior fraction of observable chimeric molecules (used for regularization)
#         e_lo_obs_prior_fraction_n = rho_ave_n * p_obs_lo_n / (rho_ave_n * p_obs_lo_n + p_obs_hi_n)
#
#         return {
#             'mu_e_lo_n': mu_e_lo_n,
#             'e_lo_obs_prior_fraction_n': e_lo_obs_prior_fraction_n
#         }
