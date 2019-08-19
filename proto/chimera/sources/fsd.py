import numpy as np
from typing import Tuple, List, Dict, Union

from pyro.distributions.torch_distribution import TorchDistribution
from boltons.cacheutils import cachedproperty

import torch
from torch.distributions import transforms
from torch.nn.parameter import Parameter

from pyro_extras import NegativeBinomial, MixtureDistribution
from fingerprint import SingleCellFingerprintDTM

from abc import abstractmethod


class FSDCodec(torch.nn.Module):
    def __init__(self):
        super(FSDCodec, self).__init__()

    @property
    @abstractmethod
    def total_fsd_params(self) -> int:
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


class NBMixtureFSDCodec(FSDCodec):
    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 n_fsd_lo_comps: int,
                 n_fsd_hi_comps: int,
                 fsd_init_params_dict: Dict[str, float],
                 device=torch.device("cuda"),
                 dtype=torch.float):
        super(NBMixtureFSDCodec, self).__init__()
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
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
        mean_fsd_mu_hi = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi)
        mean_fsd_phi_hi = np.mean(sc_fingerprint_dtm.empirical_fsd_phi_hi)
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
                           downsampling_rate_tensor: Union[None, torch.Tensor] = None) \
            -> Tuple[TorchDistribution, TorchDistribution]:
        # instantiate the "chimeric" (lo) distribution
        log_w_nb_lo_tuple = tuple(
            fsd_params_dict['w_lo'][..., j].log().unsqueeze(-1) for j in range(self.n_fsd_lo_comps))
        if downsampling_rate_tensor is None:
            nb_lo_components_tuple = tuple(NegativeBinomial(
                fsd_params_dict['mu_lo'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_lo'][..., j].unsqueeze(-1)) for j in range(self.n_fsd_lo_comps))
        else:
            nb_lo_components_tuple = tuple(NegativeBinomial(
                downsampling_rate_tensor.unsqueeze(-1) * fsd_params_dict['mu_lo'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_lo'][..., j].unsqueeze(-1)) for j in range(self.n_fsd_lo_comps))

        # instantiate the "real" (hi) distribution
        log_w_nb_hi_tuple = tuple(
            fsd_params_dict['w_hi'][..., j].log().unsqueeze(-1) for j in range(self.n_fsd_hi_comps))
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

    # TODO magic numbers
    def generate_fsd_init_params(self, mu_hi_guess, phi_hi_guess):
        mu_lo = self.fsd_init_mu_lo_to_mu_hi_ratio * mu_hi_guess * np.power(
            np.asarray([self.fsd_init_mu_decay]), np.arange(self.n_fsd_lo_comps))
        mu_lo = np.maximum(mu_lo, 1.1 * self.fsd_init_min_mu_lo)
        phi_lo = np.ones((self.n_fsd_lo_comps,))
        w_lo = np.power(np.asarray([self.fsd_init_w_decay]), np.arange(self.n_fsd_lo_comps))
        w_lo = w_lo / np.sum(w_lo)

        mu_hi = mu_hi_guess * np.power(
            np.asarray([self.fsd_init_mu_decay]), np.arange(self.n_fsd_hi_comps))
        mu_hi = np.maximum(mu_hi, 1.1 * self.fsd_init_min_mu_hi)
        phi_hi = min(phi_hi_guess, 0.9 * self.fsd_init_max_phi_hi) * np.ones((self.n_fsd_hi_comps,))
        w_hi = np.power(np.asarray([self.fsd_init_w_decay]), np.arange(self.n_fsd_hi_comps))
        w_hi = w_hi / np.sum(w_hi)

        return mu_lo, phi_lo, w_lo, mu_hi, phi_hi, w_hi

    @cachedproperty
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

    @cachedproperty
    def init_fsd_xi_loc_posterior(self):
        xi_list = []
        for i_gene in range(self.sc_fingerprint_dtm.n_genes):
            mu_lo, phi_lo, w_lo, mu_hi, phi_hi, w_hi = self.generate_fsd_init_params(
                self.sc_fingerprint_dtm.empirical_fsd_mu_hi[i_gene],
                self.sc_fingerprint_dtm.empirical_fsd_phi_hi[i_gene])
            xi = self.encode({
                'mu_lo': torch.tensor(mu_lo, dtype=self.dtype),
                'phi_lo': torch.tensor(phi_lo, dtype=self.dtype),
                'w_lo': torch.tensor(w_lo, dtype=self.dtype),
                'mu_hi': torch.tensor(mu_hi, dtype=self.dtype),
                'phi_hi': torch.tensor(phi_hi, dtype=self.dtype),
                'w_hi': torch.tensor(w_hi, dtype=self.dtype)})
            xi_list.append(xi.unsqueeze(0))
        return torch.cat(xi_list, 0).to(self.device)


class SortByComponentWeights(transforms.Transform):
    def __init__(self, fsd_codec: FSDCodec):
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
        assert (y not in self._intermediates_cache), \
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[y] = x

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x)
