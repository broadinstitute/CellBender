import numpy as np
from typing import Tuple, List, Dict, Union

from pyro.distributions.torch_distribution import TorchDistribution
from boltons.cacheutils import cachedproperty

import torch
from torch.distributions import transforms, constraints
from torch.nn.parameter import Parameter

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.gp.models import VariationalSparseGP
import pyro.contrib.gp.kernels as kernels

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


class NBMixtureFSDCodec(FSDCodec):

    stick = transforms.StickBreakingTransform()

    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 n_fsd_lo_comps: int,
                 n_fsd_hi_comps: int,
                 init_params_dict: Dict[str, float],
                 device=torch.device("cuda"),
                 dtype=torch.float):
        super(NBMixtureFSDCodec, self).__init__()
        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.n_fsd_lo_comps = n_fsd_lo_comps
        self.n_fsd_hi_comps = n_fsd_hi_comps

        self.fsd_init_min_mu_lo = init_params_dict['fsd.init_min_mu_lo']
        self.fsd_init_min_mu_hi = init_params_dict['fsd.init_min_mu_hi']
        self.fsd_init_max_phi_lo = init_params_dict['fsd.init_max_phi_lo']
        self.fsd_init_max_phi_hi = init_params_dict['fsd.init_max_phi_hi']
        self.fsd_init_mu_decay = init_params_dict['fsd.init_mu_decay']
        self.fsd_init_w_decay = init_params_dict['fsd.init_w_decay']
        self.fsd_init_mu_lo_to_mu_hi_ratio = init_params_dict['fsd.init_mu_lo_to_mu_hi_ratio']

        self.device = device
        self.dtype = dtype

        # initialization of p_lo and p_hi
        mean_fsd_mu_hi = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi)
        mean_fsd_phi_hi = np.mean(sc_fingerprint_dtm.empirical_fsd_phi_hi)
        (self.init_fsd_mu_lo, self.init_fsd_phi_lo, self.init_fsd_w_lo,
         self.init_fsd_mu_hi, self.init_fsd_phi_hi, self.init_fsd_w_hi) = self.generate_fsd_init_params(
            mean_fsd_mu_hi, mean_fsd_phi_hi)

    @property
    def total_fsd_params(self):
        n_lo = 3 * self.n_fsd_lo_comps - 1
        n_hi = 3 * self.n_fsd_hi_comps - 1
        return n_lo + n_hi

    @staticmethod
    def decode_xi(fsd_xi: torch.Tensor,
                  n_fsd_lo_comps: int,
                  n_fsd_hi_comps: int) -> Dict[str, torch.Tensor]:
        n_lo = 3 * n_fsd_lo_comps - 1
        n_hi = 3 * n_fsd_hi_comps - 1
        assert fsd_xi.shape[-1] == (n_lo + n_hi)
        offset = 0

        # p_hi parameters are directly transformed from fsd_xi
        log_mu_hi = fsd_xi[..., offset:(offset + n_fsd_hi_comps)]
        mu_hi = log_mu_hi.exp()
        offset += n_fsd_hi_comps

        log_phi_hi = fsd_xi[..., offset:(offset + n_fsd_hi_comps)]
        phi_hi = log_phi_hi.exp()
        offset += n_fsd_hi_comps

        if n_fsd_hi_comps > 1:
            w_hi = NBMixtureFSDCodec.stick(fsd_xi[..., offset:(offset + n_fsd_hi_comps - 1)])
            offset += (n_fsd_hi_comps - 1)
        else:
            w_hi = torch.ones_like(mu_hi)

        # p_lo parameters are directly transformed from fsd_xi
        log_mu_lo = fsd_xi[..., offset:(offset + n_fsd_lo_comps)]
        mu_lo = log_mu_lo.exp()
        offset += n_fsd_lo_comps

        log_phi_lo = fsd_xi[..., offset:(offset + n_fsd_lo_comps)]
        phi_lo = log_phi_lo.exp()
        offset += n_fsd_lo_comps

        if n_fsd_lo_comps > 1:
            w_lo = NBMixtureFSDCodec.stick(fsd_xi[..., offset:(offset + n_fsd_lo_comps - 1)])
            offset += (n_fsd_lo_comps - 1)
        else:
            w_lo = torch.ones_like(mu_lo)

        return {'mu_lo': mu_lo,
                'phi_lo': phi_lo,
                'w_lo': w_lo,
                'mu_hi': mu_hi,
                'phi_hi': phi_hi,
                'w_hi': w_hi}

    def decode(self, fsd_xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decode_xi(
            fsd_xi=fsd_xi,
            n_fsd_lo_comps=self.n_fsd_lo_comps,
            n_fsd_hi_comps=self.n_fsd_hi_comps)

    @staticmethod
    def encode_xi(fsd_params_dict: Dict[str, torch.Tensor],
                  n_fsd_lo_comps: int,
                  n_fsd_hi_comps: int) -> torch.Tensor:
        xi_tuple = tuple()
        xi_tuple += (fsd_params_dict['mu_hi'].log(),)
        xi_tuple += (fsd_params_dict['phi_hi'].log(),)
        if n_fsd_hi_comps > 1:
            xi_tuple += (NBMixtureFSDCodec.stick.inv(fsd_params_dict['w_hi']),)
        xi_tuple += (fsd_params_dict['mu_lo'].log(),)
        xi_tuple += (fsd_params_dict['phi_lo'].log(),)
        if n_fsd_lo_comps > 1:
            xi_tuple += (NBMixtureFSDCodec.stick.inv(fsd_params_dict['w_lo']),)
        return torch.cat(xi_tuple, -1)

    def encode(self, fsd_params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_xi(
            fsd_params_dict=fsd_params_dict,
            n_fsd_lo_comps=self.n_fsd_lo_comps,
            n_fsd_hi_comps=self.n_fsd_hi_comps)

    @staticmethod
    def get_sorted_params_dict(fsd_params_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    def get_sorted_fsd_xi(self, fsd_xi: torch.Tensor) -> torch.Tensor:
        return self.encode(self.get_sorted_params_dict(self.decode(fsd_xi)))

    @staticmethod
    def get_fsd_components_nb_mixture(
            fsd_params_dict: Dict[str, torch.Tensor],
            n_fsd_lo_comps: int,
            n_fsd_hi_comps: int,
            downsampling_rate_tensor: Union[None, torch.Tensor] = None) \
            -> Tuple[TorchDistribution, TorchDistribution]:
        # instantiate the "chimeric" (lo) distribution
        log_w_nb_lo_tuple = tuple(
            fsd_params_dict['w_lo'][..., j].log().unsqueeze(-1) for j in range(n_fsd_lo_comps))
        if downsampling_rate_tensor is None:
            nb_lo_components_tuple = tuple(NegativeBinomial(
                fsd_params_dict['mu_lo'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_lo'][..., j].unsqueeze(-1)) for j in range(n_fsd_lo_comps))
        else:
            nb_lo_components_tuple = tuple(NegativeBinomial(
                downsampling_rate_tensor.unsqueeze(-1) * fsd_params_dict['mu_lo'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_lo'][..., j].unsqueeze(-1)) for j in range(n_fsd_lo_comps))

        # instantiate the "real" (hi) distribution
        log_w_nb_hi_tuple = tuple(
            fsd_params_dict['w_hi'][..., j].log().unsqueeze(-1) for j in range(n_fsd_hi_comps))
        if downsampling_rate_tensor is None:
            nb_hi_components_tuple = tuple(NegativeBinomial(
                fsd_params_dict['mu_hi'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_hi'][..., j].unsqueeze(-1)) for j in range(n_fsd_hi_comps))
        else:
            nb_hi_components_tuple = tuple(NegativeBinomial(
                downsampling_rate_tensor.unsqueeze(-1) * fsd_params_dict['mu_hi'][..., j].unsqueeze(-1),
                fsd_params_dict['phi_hi'][..., j].unsqueeze(-1)) for j in range(n_fsd_hi_comps))

        dist_lo = MixtureDistribution(log_w_nb_lo_tuple, nb_lo_components_tuple)
        dist_hi = MixtureDistribution(log_w_nb_hi_tuple, nb_hi_components_tuple)

        return dist_lo, dist_hi

    def get_fsd_components(self,
                           fsd_params_dict: Dict[str, torch.Tensor],
                           downsampling_rate_tensor: Union[None, torch.Tensor]) \
            -> Tuple[TorchDistribution, TorchDistribution]:
        return self.get_fsd_components_nb_mixture(
            fsd_params_dict=fsd_params_dict,
            n_fsd_lo_comps=self.n_fsd_lo_comps,
            n_fsd_hi_comps=self.n_fsd_hi_comps,
            downsampling_rate_tensor=downsampling_rate_tensor)

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


class NBMixtureRealVSGPChimeraFSDCodec(FSDCodec):
    """NB mixture for real components, VSGP from real for chimeric component."""
    def __init__(self,
                 sc_fingerprint_dtm: SingleCellFingerprintDTM,
                 n_fsd_hi_comps: int,
                 init_params_dict: Dict[str, float],
                 device: torch.device = torch.device("cuda"),
                 dtype: torch.dtype = torch.float):
        super(FSDCodec, self).__init__()

        self.sc_fingerprint_dtm = sc_fingerprint_dtm
        self.n_fsd_hi_comps = n_fsd_hi_comps

        self.fsd_init_min_mu_lo = init_params_dict['fsd.init_min_mu_lo']
        self.fsd_init_min_mu_hi = init_params_dict['fsd.init_min_mu_hi']
        self.fsd_init_max_phi_lo = init_params_dict['fsd.init_max_phi_lo']
        self.fsd_init_max_phi_hi = init_params_dict['fsd.init_max_phi_hi']
        self.fsd_init_mu_decay = init_params_dict['fsd.init_mu_decay']
        self.fsd_init_w_decay = init_params_dict['fsd.init_w_decay']
        self.fsd_init_mu_lo_to_mu_hi_ratio = init_params_dict['fsd.init_mu_lo_to_mu_hi_ratio']

        self.device = device
        self.dtype = dtype

        # initialization of p_lo and p_hi
        mean_fsd_mu_hi = np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi)
        mean_fsd_phi_hi = np.mean(sc_fingerprint_dtm.empirical_fsd_phi_hi)
        (self.init_fsd_mu_lo, self.init_fsd_phi_lo, self.init_fsd_w_lo,
         self.init_fsd_mu_hi, self.init_fsd_phi_hi, self.init_fsd_w_hi) = self.generate_fsd_init_params(
            mean_fsd_mu_hi, mean_fsd_phi_hi)

        # fsd_lo GP inducing points
        self.fsd_lo_inducing_points = torch.linspace(
            np.log(np.min(sc_fingerprint_dtm.empirical_fsd_mu_hi)),
            np.log(np.max(sc_fingerprint_dtm.empirical_fsd_mu_hi)),
            steps=int(init_params_dict['fsd.vsgp.n_inducing_points']),
            device=device, dtype=dtype)

        # fsd_lo GP kernel setup
        input_dim = 1

        kernel_rbf = kernels.RBF(
            input_dim=input_dim,
            variance=torch.tensor(
                init_params_dict['fsd.vsgp.init_rbf_kernel_variance'], device=device, dtype=dtype),
            lengthscale=torch.tensor(
                init_params_dict['fsd.vsgp.init_rbf_kernel_lengthscale'], device=device, dtype=dtype))

        kernel_linear = kernels.Linear(
            input_dim=input_dim,
            variance=torch.tensor(
                init_params_dict['fsd.vsgp.init_linear_kernel_variance'], device=device, dtype=dtype))

        kernel_constant = kernels.Constant(
            input_dim=input_dim,
            variance=torch.tensor(
                init_params_dict['fsd.vsgp.init_constant_kernel_variance'], device=device, dtype=dtype))

        kernel_whitenoise = kernels.WhiteNoise(
            input_dim=input_dim,
            variance=torch.tensor(
                init_params_dict['fsd.vsgp.init_whitenoise_kernel_variance'], device=device, dtype=dtype))
        kernel_whitenoise.set_constraint(
            "variance", constraints.greater_than(init_params_dict['fsd.vsgp.min_noise']))

        kernel_full = kernels.Sum(
            kernel_rbf,
            kernels.Sum(
                kernel_linear,
                kernels.Sum(
                    kernel_whitenoise,
                    kernel_constant)))

        # mean subtraction
        self.log_mu_lo_mean = torch.nn.Parameter(
            torch.tensor(
                [np.log(init_params_dict['fsd.init_mu_lo_to_mu_hi_ratio']
                        * np.mean(sc_fingerprint_dtm.empirical_fsd_mu_hi))],
                device=device, dtype=dtype).unsqueeze(-1))

        # instantiate VSGP model
        self.vsgp = VariationalSparseGP(
            X=None,
            y=None,
            kernel=kernel_full,
            Xu=self.inducing_points,
            likelihood=None,
            mean_function=lambda x: self.log_mu_lo_mean,
            latent_shape=torch.Size([1]),
            whiten=True,
            jitter=init_params_dict['fsd.vsgp.cholesky_jitter'])

        # send parameters to device
        self.to(device)

    @property
    def total_fsd_params(self):
        return 3 * self.n_fsd_hi_comps - 1

    def decode_xi_to_fsd_hi_params_dict(self, fsd_xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert fsd_xi.shape[-1] == (3 * self.n_fsd_hi_comps - 1)
        offset = 0

        # p_hi parameters are directly transformed from fsd_xi
        log_mu_hi = fsd_xi[..., offset:(offset + self.n_fsd_hi_comps)]
        mu_hi = log_mu_hi.exp()
        offset += self.n_fsd_hi_comps

        log_phi_hi = fsd_xi[..., offset:(offset + self.n_fsd_hi_comps)]
        phi_hi = log_phi_hi.exp()
        offset += self.n_fsd_hi_comps

        if self.n_fsd_hi_comps > 1:
            w_hi = NBMixtureFSDCodec.stick(fsd_xi[..., offset:(offset + self.n_fsd_hi_comps - 1)])
            offset += (self.n_fsd_hi_comps - 1)
        else:
            w_hi = torch.ones_like(mu_hi)

        return {'mu_hi': mu_hi,
                'phi_hi': phi_hi,
                'w_hi': w_hi}

    def sample_fsd_lo_params_dict(self, fsd_hi_params_dist: Dict[str, torch.Tensor]) \
            -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def decode(self, fsd_xi: torch.Tensor) -> Dict[str, torch.Tensor]:
        fsd_hi_params_dict = self.decode_xi_to_fsd_hi_params_dict(fsd_xi)
        fsd_lo_params_dict = self.sample_fsd_lo_params_dict(fsd_hi_params_dict)
        return {**fsd_lo_params_dict, **fsd_hi_params_dict}

    def encode(self, fsd_params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        xi_tuple = tuple()
        xi_tuple += (fsd_params_dict['mu_hi'].log(),)
        xi_tuple += (fsd_params_dict['phi_hi'].log(),)
        if self.n_fsd_hi_comps > 1:
            xi_tuple += (NBMixtureFSDCodec.stick.inv(fsd_params_dict['w_hi']),)
        return torch.cat(xi_tuple, -1)

    @staticmethod
    def get_sorted_fsd_hi_params_dict(fsd_hi_params_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fsd_hi_sort_order = torch.argsort(fsd_hi_params_dict['w_hi'], dim=-1, descending=True)
        sorted_fsd_hi_params_dict = {
            'mu_hi': torch.gather(fsd_hi_params_dict['mu_hi'], dim=-1, index=fsd_hi_sort_order),
            'phi_hi': torch.gather(fsd_hi_params_dict['phi_hi'], dim=-1, index=fsd_hi_sort_order),
            'w_hi': torch.gather(fsd_hi_params_dict['w_hi'], dim=-1, index=fsd_hi_sort_order)}
        return sorted_fsd_hi_params_dict

    def get_sorted_fsd_xi(self, fsd_xi: torch.Tensor) -> torch.Tensor:
        return self.encode(
            self.get_sorted_fsd_hi_params_dict(
                self.decode_xi_to_fsd_hi_params_dict(fsd_xi)))

    def get_fsd_components(self,
                           fsd_params_dict: Dict[str, torch.Tensor],
                           downsampling_rate_tensor: Union[None, torch.Tensor] = None) \
            -> Tuple[TorchDistribution, TorchDistribution]:
        return NBMixtureFSDCodec.get_fsd_components_nb_mixture(
            fsd_params_dict=fsd_params_dict,
            n_fsd_lo_comps=1,
            n_fsd_hi_comps=self.n_fsd_hi_comps,
            downsampling_rate_tensor=downsampling_rate_tensor)

    def generate_fsd_init_params(self, mu_hi_guess, phi_hi_guess):
        mu_hi = mu_hi_guess * np.power(
            np.asarray([self.fsd_init_mu_decay]), np.arange(self.n_fsd_hi_comps))
        mu_hi = np.maximum(mu_hi, 1.1 * self.fsd_init_min_mu_hi)
        phi_hi = min(phi_hi_guess, 0.9 * self.fsd_init_max_phi_hi) * np.ones((self.n_fsd_hi_comps,))
        w_hi = np.power(np.asarray([self.fsd_init_w_decay]), np.arange(self.n_fsd_hi_comps))
        w_hi = w_hi / np.sum(w_hi)

        return mu_hi, phi_hi, w_hi

    @cachedproperty
    def init_fsd_xi_loc_prior(self):
        mu_hi = torch.tensor(self.init_fsd_mu_hi, device=self.device, dtype=self.dtype)
        phi_hi = torch.tensor(self.init_fsd_phi_hi, device=self.device, dtype=self.dtype)
        w_hi = torch.tensor(self.init_fsd_w_hi, device=self.device, dtype=self.dtype)

        return self.encode({
            'mu_hi': mu_hi,
            'phi_hi': phi_hi,
            'w_hi': w_hi})

    @cachedproperty
    def init_fsd_xi_loc_posterior(self):
        xi_list = []
        for i_gene in range(self.sc_fingerprint_dtm.n_genes):
            mu_hi, phi_hi, w_hi = self.generate_fsd_init_params(
                self.sc_fingerprint_dtm.empirical_fsd_mu_hi[i_gene],
                self.sc_fingerprint_dtm.empirical_fsd_phi_hi[i_gene])
            xi = self.encode({
                'mu_hi': torch.tensor(mu_hi, dtype=self.dtype),
                'phi_hi': torch.tensor(phi_hi, dtype=self.dtype),
                'w_hi': torch.tensor(w_hi, dtype=self.dtype)})
            xi_list.append(xi.unsqueeze(0))
        return torch.cat(xi_list, 0).to(self.device)
