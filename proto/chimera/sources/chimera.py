import numpy as np
from typing import Tuple, List, Dict, Union, Any, Callable, Generator, Optional
from abc import abstractmethod
import logging
import time

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
from pyro.contrib.gp.parameterized import Parameterized, Parameter

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
            loc=init_params_dict['chimera_alpha_c_prior_loc'],
            scale=init_params_dict['chimera_alpha_c_prior_scale'])

        self.beta_c_prior_a, self.beta_c_prior_b = gamma_loc_scale_to_concentration_rate(
            loc=init_params_dict['chimera_beta_c_prior_loc'],
            scale=init_params_dict['chimera_beta_c_prior_scale'])

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
        self.alpha_c_posterior_loc = Parameter(
            torch.tensor(self.alpha_c_prior_a / self.alpha_c_prior_b, device=self.device, dtype=self.dtype))
        self.set_constraint("alpha_c_posterior_loc", constraints.positive)

        self.beta_c_posterior_loc = Parameter(
            torch.tensor(self.beta_c_prior_a / self.beta_c_prior_b, device=self.device, dtype=self.dtype))
        self.set_constraint("beta_c_posterior_loc", constraints.positive)

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
        assert 'total_obs_gene_expr_per_cell_n' in parents_dict
        assert 'p_obs_lo_n' in parents_dict
        assert 'p_obs_hi_n' in parents_dict

        alpha_c = output_dict['alpha_c']
        beta_c = output_dict['beta_c']

        mu_fsd_hi_n = parents_dict['mu_fsd_hi_n']
        eta_n = parents_dict['eta_n']
        total_obs_gene_expr_per_cell_n = parents_dict['total_obs_gene_expr_per_cell_n']
        p_obs_lo_n = parents_dict['p_obs_lo_n']
        p_obs_hi_n = parents_dict['p_obs_hi_n']

        # calculate chimera rate
        scaled_mu_fsd_hi_n = mu_fsd_hi_n / self.mean_empirical_fsd_mu_hi
        rho_n = (alpha_c + beta_c * eta_n) * scaled_mu_fsd_hi_n
        rho_ave_n = (alpha_c + beta_c) * scaled_mu_fsd_hi_n
        total_fragments_n = total_obs_gene_expr_per_cell_n / (rho_ave_n * p_obs_lo_n + p_obs_hi_n)
        mu_e_lo_n = rho_n * total_fragments_n

        # prior fraction of observable chimeric molecules (used for regularization)
        e_lo_obs_prior_fraction_n = rho_ave_n * p_obs_lo_n / (rho_ave_n * p_obs_lo_n + p_obs_hi_n)

        return {
            'mu_e_lo_n': mu_e_lo_n,
            'e_lo_obs_prior_fraction_n': e_lo_obs_prior_fraction_n
        }
