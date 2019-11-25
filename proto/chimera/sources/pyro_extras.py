import os
import numpy as np
import pandas as pd
import itertools
import operator
import logging

from typing import List

import pyro
from pyro.distributions.torch_distribution import TorchDistribution, TorchDistributionMixin
from pyro.distributions.util import broadcast_shape
from pyro.contrib.gp.kernels import Kernel
from pyro.nn.module import PyroParam

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs, broadcast_all, lazy_property
from torch.distributions import constraints, transforms
from typing import Tuple
from numbers import Number

from torch._six import nan


def get_kl_divergence(log_p, log_q):
    out = - log_p.exp() * (log_q - log_p)
    out[torch.isnan(out) | torch.isinf(out)] = 0
    return out


def get_js_divergence(log_p, log_q):
    return 0.25 * (get_kl_divergence(log_p, log_q) + get_kl_divergence(log_q, log_p))


def get_hellinger_distance(log_p, log_q, reduce=None, log_prob_min=-50.0):
    prod = (log_p.exp() * log_q.exp()).clone()
    prod[log_p < log_prob_min] = 0
    prod[log_q < log_prob_min] = 0
    if reduce is not None:
        return 1 - reduce(prod.sqrt())
    else:
        return 1 - prod.sqrt().sum()

def get_bhattacharyya_distance(log_p, log_q, reduce=None, log_safeguard_eps=1e-12):
    prod = (log_p.exp() * log_q.exp()).sqrt()
    if reduce is not None:
        bc = reduce(prod)
    else:
        bc = prod.sum()
    return - (bc + log_safeguard_eps).log() + np.log(log_safeguard_eps)

    
def logit(x: torch.Tensor):
    return torch.log(x * (1 - x).reciprocal())


def logaddexp(a: torch.Tensor, b: torch.Tensor):
    a, b = broadcast_all(a, b)
    return torch.stack((a, b), -1).logsumexp(-1)


LN_1_M_EXP_THRESHOLD = -np.log(2.)


def get_log_prob_compl(log_prob: torch.Tensor):
    return torch.where(
        log_prob >= LN_1_M_EXP_THRESHOLD,
        torch.log(-torch.expm1(log_prob)),
        torch.log1p(-torch.exp(log_prob)))


class CustomLogProbTerm(TorchDistribution):
    def __init__(self, custom_log_prob, batch_shape, event_shape, validate_args=None):
        self.custom_log_prob = custom_log_prob
        super(CustomLogProbTerm, self).__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CustomLogProbTerm, _instance)
        batch_shape = torch.Size(batch_shape)
        new.custom_log_prob = self.custom_log_prob.expand(batch_shape + self.event_shape)
        super(CustomLogProbTerm, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        return self.custom_log_prob
    

class NegativeBinomial(TorchDistribution):
    r"""
    Creates a negative binomial distribution.
    
    Args:
        mu (Number, Tensor): mean (must be strictly positive)
        phi (Number, Tensor): overdispersion (must be strictly positive)
    """
    arg_constraints = {'mu': constraints.positive, 'phi': constraints.positive}
    support = constraints.positive_integer
    EPS = 1e-6
    
    def __init__(self, mu, phi, validate_args=None):
        self.mu, self.phi = broadcast_all(mu, phi)
        if all(isinstance(_var, Number) for _var in (mu, phi)): 
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(NegativeBinomial, self).__init__(batch_shape, validate_args=validate_args)
        
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.phi = self.phi.expand(batch_shape)
        
        super(NegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    @lazy_property
    def _gamma(self):
        return torch.distributions.Gamma(
            concentration=self.phi.reciprocal(),
            rate=(self.mu * self.phi).reciprocal())

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return torch.poisson(self._gamma.sample(sample_shape=sample_shape))
        
    @property
    def mean(self):
        return self.mu
    
    @property
    def variance(self):
        return self.mu + self.phi * self.mu.pow(2)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        mu, phi, value = broadcast_all(self.mu, self.phi, value)
        alpha = (self.EPS + phi).reciprocal()
        return ((value + alpha).lgamma() - (value + 1).lgamma() - alpha.lgamma()
                + alpha * (alpha.log() - (alpha + mu).log())
                + value * (mu.log() - (alpha + mu).log()))
  
    
class ZeroInflatedNegativeBinomial(TorchDistribution):
    r"""
    Creates a negative binomial distribution.
    
    Args:
        logit_p_zero (Number, Tensor): zero inflation probability in logit scale
        mu (Number, Tensor): mean (must be strictly positive)
        phi (Number, Tensor): overdispersion (must be strictly positive)
    """
    arg_constraints = {
        'logit_p_zero': constraints.real,
        'mu': constraints.positive,
        'phi': constraints.positive}
    
    support = constraints.positive_integer
    EPS = 1e-6
    
    def __init__(self, logit_p_zero, mu, phi, validate_args=None):
        self.logit_p_zero, self.mu, self.phi = broadcast_all(logit_p_zero, mu, phi)
        if all(isinstance(_var, Number) for _var in (logit_p_zero, mu, phi)):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(ZeroInflatedNegativeBinomial, self).__init__(batch_shape, validate_args=validate_args)
        
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ZeroInflatedNegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.logit_p_zero = self.logit_p_zero.expand(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.phi = self.phi.expand(batch_shape)
        
        super(ZeroInflatedNegativeBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    @lazy_property
    def prob_zero(self):
        return torch.sigmoid(self.logit_p_zero)
    
    @lazy_property
    def log_prob_zero(self):
        return torch.nn.functional.logsigmoid(self.logit_p_zero)
    
    @lazy_property
    def log_prob_nonzero(self):
        return get_log_prob_compl(self.log_prob_zero)

    @lazy_property
    def _gamma(self):
        return torch.distributions.Gamma(
            concentration=self.phi.reciprocal(),
            rate=(self.mu * self.phi).reciprocal())

    @lazy_property
    def _bernoulli(self):
        return torch.distributions.Bernoulli(logits=self.logit_p_zero)
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            mask = self._bernoulli.sample(sample_shape=sample_shape)
            nb = torch.poisson(self._gamma.sample(sample_shape=sample_shape))
            return (1. - mask) * nb
        
    @property
    def mean(self):
        return (1. - self.prob_zero) * self.mu
    
    @property
    def variance(self):
        return (1. - self.prob_zero) * self.mu * (1. + self.mu * (self.prob_zero + self.phi))
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob_zero, log_prob_nonzero, mu, phi, value = broadcast_all(
            self.log_prob_zero, self.log_prob_nonzero, self.mu, self.phi, value)
        z_mask = (value == 0)
        nnz_mask = torch.bitwise_not(z_mask)
        out = torch.zeros_like(value)
        alpha = (self.EPS + phi).reciprocal()
        log_prob_nb_zero = alpha * (alpha.log() - (alpha + mu).log())
        out[z_mask] = logaddexp(log_prob_zero, log_prob_nonzero + log_prob_nb_zero)[z_mask]
        out[nnz_mask] = (
            log_prob_nonzero + (
                (value + alpha).lgamma() - (value + 1).lgamma() - alpha.lgamma() +
                alpha * (alpha.log() - (alpha + mu).log()) +
                value * (mu.log() - (alpha + mu).log())))[nnz_mask]
        return out
    
    
class MixtureConstraint(constraints.Constraint):
    def __init__(self, constraints: List[constraints.Constraint]):
        self.constraints = constraints

    def check(self, value):
        result = self.constraints[0].check(value)
        for constraint in self.constraints[1:]:
            result = result & constraint.check(value)
        return result


class MixtureDistribution(TorchDistribution):
    arg_constraints = {}  # nothing can be constrained

    def __init__(self, log_weights: Tuple[torch.Tensor], components: Tuple[TorchDistribution],
                 normalize_weights=False, validate_args=None):
        # basic sanity checks
        assert isinstance(log_weights, tuple)
        assert isinstance(components, tuple)
        assert len(log_weights) > 0
        assert len(log_weights) == len(components), \
            f'List of weights and components must have equal lengths: {len(log_weights)} vs {len(components)}'
        
        # all components must have the same event_shape
        event_shape = components[0].event_shape
        for component in components[1:]:
            assert component.event_shape == event_shape, \
                f'Components event_shape disagree: {component.event_shape} vs {event_shape}'
        
        # broadcast batch shapes across weights and components
        batch_shape = broadcast_shape(
            *(log_weight.shape for log_weight in log_weights),
            *(component.batch_shape for component in components))
        self.log_weights = tuple(log_weight.expand(batch_shape) if log_weight.shape != batch_shape else log_weight
                                 for log_weight in log_weights)
        self.components = tuple(component.expand(batch_shape) if component.batch_shape != batch_shape else component
                                for component in components)

        if normalize_weights:
            log_norm = torch.logsumexp(torch.cat(tuple(log_weight.unsqueeze(-1) for log_weight in self.log_weights), -1), -1)
            self.log_weights = tuple(log_weight - log_norm for log_weight in self.log_weights)

        super(MixtureDistribution, self).__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        first_component_support = self.components[0].support
        if all(component.support == first_component_support for component in self.components):
            return first_component_support
        return MixtureConstraint([component.support for component in self.components])

    def expand(self, batch_shape):
        log_weights = tuple(log_weight.expand(batch_shape) for log_weight in self.log_weights)
        components = tuple(component.expand(batch_shape) for component in self.components)
        return type(self)(log_weights, components)

    @property
    def mean(self):
        return torch.sum(
            torch.cat(
                tuple((log_weight.exp() * component.mean).unsqueeze(-1)
                      for log_weight, component in zip(self.log_weights, self.components)),
                -1),
            -1)
    
    @lazy_property
    def stacked_weights(self):
        return torch.cat(tuple(log_weight.exp().unsqueeze(-1) for log_weight in self.log_weights), -1)
        
    def sample(self, sample_shape=torch.Size([])):
        assignments = torch.distributions.Categorical(
            probs=self.stacked_weights).sample(sample_shape)
        expanded_assignments = assignments.view(
            assignments.shape + (1,) *len(self.event_shape)).expand(assignments.shape + self.event_shape)
        component_samples = torch.stack(
            tuple(component.sample(sample_shape) for component in self.components), -1)
        return torch.gather(component_samples, -1, expanded_assignments.unsqueeze(-1)).squeeze(-1)
    
    def log_prob(self, value):
        value_shape = broadcast_shape(value.shape, self.batch_shape + self.event_shape)
        if value.shape != value_shape:
            value = value.expand(value_shape)
        if self._validate_args:
            self._validate_sample(value)
        weight_shape = value_shape[:len(value_shape) - len(self.event_shape)]
        log_weights = tuple(log_weight.expand(weight_shape) if log_weight.shape != weight_shape else log_weight
                            for log_weight in self.log_weights)
        log_probs = (component.log_prob(value) for component in self.components)
        return torch.logsumexp(torch.cat(tuple(
            log_prob.unsqueeze(-1) + log_weight.unsqueeze(-1)
            for log_prob, log_weight in zip(log_probs, log_weights)), -1), -1)


class WhiteNoiseWithMinVariance(Kernel):
    r"""
    Implementation of WhiteNoise kernel:

        :math:`k(x, z) = \sigma^2 \delta(x, z),`

    where :math:`\delta` is a Dirac delta function.
    """

    def __init__(self, input_dim, variance=None, active_dims=None, min_noise=None):
        super(WhiteNoiseWithMinVariance, self).__init__(input_dim, active_dims)

        variance = torch.tensor(1.) if variance is None else variance
        if min_noise:
            self.variance = PyroParam(variance, constraints.greater_than(min_noise))
        else:
            self.variance = PyroParam(variance, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            return self.variance.expand(X.size(0)).diag()
        else:
            return X.data.new_zeros(X.size(0), Z.size(0))


def get_binomial_samples_sparse_counts(
        total_counts: torch.Tensor,
        logits: torch.Tensor,
        sample_shape: torch.Size):
    total_counts_x, logits_x = broadcast_all(total_counts, logits)
    total_counts_n = total_counts_x.flatten()
    total_counts_nnz_mask_n = total_counts_n > 0
    if torch.any(total_counts_nnz_mask_n).item():
        total_counts_nnz_m = total_counts_n[total_counts_nnz_mask_n]
        logits_nnz_m = logits_x.flatten()[total_counts_nnz_mask_n]
        binom_nnz_samples_sm = torch.distributions.Binomial(
            total_count=total_counts_nnz_m,
            logits=logits_nnz_m).sample(sample_shape)
        binom_samples_sn = torch.zeros(
            sample_shape + total_counts_n.shape,
            dtype=total_counts.dtype, device=total_counts.device)
        binom_samples_sn[..., total_counts_nnz_mask_n] = binom_nnz_samples_sm
        return binom_samples_sn.view(sample_shape + total_counts_x.shape)
    else:
        return torch.zeros(
            sample_shape + total_counts_x.shape,
            dtype=total_counts.dtype, device=total_counts.device)


def get_confidence_interval(cdf: torch.Tensor, lower_cdf: float, upper_cdf: float):
    """Calculates confidence intervals from a given empirical CDF along axis=0. A single batch dimension
    is expected at axis=1.
    
    Note:
        The CDF must be properly normalized to 1.0. This is not asserted for speed.    
    """
    assert lower_cdf >= 0.
    assert upper_cdf <= 1.
    assert lower_cdf < upper_cdf
    assert cdf.dim() == 2
    
    ladder = 0.5 * torch.linspace(0, 1, steps=cdf.shape[0], device=cdf.device, dtype=cdf.dtype)
    flipped_ladder = torch.flip(ladder, (0,))
    lo_mask = cdf < lower_cdf
    hi_mask = cdf > upper_cdf
    lo_mask_non_degenerate = lo_mask.float() + ladder.unsqueeze(-1)
    hi_mask_non_degenerate = hi_mask.float() + flipped_ladder.unsqueeze(-1)
    lo_idx = torch.any(lo_mask, dim=0).long() * lo_mask_non_degenerate.argmax(dim=0)
    hi_idx = torch.any(1 - hi_mask, dim=0).long() * torch.clamp(
        hi_mask_non_degenerate.argmin(dim=0) + 1, max=cdf.shape[0] - 1)
    return lo_idx, hi_idx


def generate_next_checkpoint_path(checkpoint_path: str, prefix: str = 'model_checkpoint') -> str:
    try:
        os.mkdir(checkpoint_path)
    except:
        pass
    n = 0
    while True:
        candidate_checkpoint_path = os.path.join(checkpoint_path, f'{prefix}_{n:05d}.pyro')
        if os.path.exists(candidate_checkpoint_path):
            n += 1
            continue
        break
    return candidate_checkpoint_path


def checkpoint_model(checkpoint_path: str, prefix: str = 'model_checkpoint'):
    pyro.get_param_store().save(generate_next_checkpoint_path(checkpoint_path, prefix))


def load_latest_checkpoint(checkpoint_path: str, prefix: str = 'model_checkpoint'):
    n = 0
    last_existing_checkpoint_path = None
    while True:
        candidate_checkpoint_path = os.path.join(checkpoint_path, f'{prefix}_{n:05d}.pyro')
        if os.path.exists(candidate_checkpoint_path):
            last_existing_checkpoint_path = candidate_checkpoint_path
            n += 1
            continue
        break
    logging.warning(f'Loading the latest available parameter checkpoint from {last_existing_checkpoint_path}...')
    pyro.get_param_store().load(last_existing_checkpoint_path)
