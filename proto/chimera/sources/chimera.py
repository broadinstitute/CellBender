import numpy as np
from typing import Dict
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
from utils import get_cell_averaged_from_collapsed_samples, get_detached_on_non_inducing_genes
import consts


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
        self._eps = 1e-7

        self.alpha_c_prior_a, self.alpha_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['chimera.alpha_c_prior_loc'],
            scale=init_params_dict['chimera.alpha_c_prior_scale'])

        self.beta_c_prior_a, self.beta_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['chimera.beta_c_prior_loc'],
            scale=init_params_dict['chimera.beta_c_prior_scale'])

        self.detach_non_inducing_genes = init_params_dict['chimera.detach_non_inducing_genes']

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

        assert 'mu_e_hi_n' in parents_dict
        assert 'gene_index_tensor_n' in parents_dict
        assert 'cell_sampling_site_scale_factor_tensor_n' in parents_dict
        assert 'mu_fsd_hi_n' in parents_dict
        assert 'total_obs_molecules_per_cell_tensor_n' in parents_dict

        alpha_c = output_dict['alpha_c']
        beta_c = output_dict['beta_c']

        if self.detach_non_inducing_genes:

            assert 'inducing_binary_mask_tensor_n' in parents_dict
            assert 'non_inducing_binary_mask_tensor_n' in parents_dict

            inducing_binary_mask_tensor_n = parents_dict['inducing_binary_mask_tensor_n']
            non_inducing_binary_mask_tensor_n = parents_dict['non_inducing_binary_mask_tensor_n']

            alpha_c_n = get_detached_on_non_inducing_genes(
                input_scalar=alpha_c,
                inducing_binary_mask_tensor_n=inducing_binary_mask_tensor_n,
                non_inducing_binary_mask_tensor_n=non_inducing_binary_mask_tensor_n)

            beta_c_n = get_detached_on_non_inducing_genes(
                input_scalar=beta_c,
                inducing_binary_mask_tensor_n=inducing_binary_mask_tensor_n,
                non_inducing_binary_mask_tensor_n=non_inducing_binary_mask_tensor_n)

        else:

            alpha_c_n = alpha_c
            beta_c_n = beta_c

        mu_e_hi_n = parents_dict['mu_e_hi_n']
        gene_index_tensor_n = parents_dict['gene_index_tensor_n']
        cell_sampling_site_scale_factor_tensor_n = parents_dict['cell_sampling_site_scale_factor_tensor_n']
        mu_fsd_hi_n = parents_dict['mu_fsd_hi_n']
        total_obs_molecules_per_cell_tensor_n = parents_dict['total_obs_molecules_per_cell_tensor_n']
        norm_total_counts_n = total_obs_molecules_per_cell_tensor_n / consts.TOTAL_COUNT_NORM_SCALE

        # auxiliary
        total_fragments_scaled_n = mu_e_hi_n * mu_fsd_hi_n / self.mean_empirical_fsd_mu_hi

        # calculate the required cell-averaged quantities
        total_fragments_scaled_cell_averaged_n = get_cell_averaged_from_collapsed_samples(
            input_tensor_n=total_fragments_scaled_n,
            gene_index_tensor_n=gene_index_tensor_n,
            cell_sampling_site_scale_factor_tensor_n=cell_sampling_site_scale_factor_tensor_n,
            n_genes=self.sc_fingerprint_dtm.n_genes,
            dtype=self.dtype,
            device=self.device)

        # calculate per-cell chimera rate
        mu_e_lo_n = (alpha_c_n + beta_c_n * norm_total_counts_n) * total_fragments_scaled_cell_averaged_n

        return {
            'mu_e_lo_n': mu_e_lo_n
        }
