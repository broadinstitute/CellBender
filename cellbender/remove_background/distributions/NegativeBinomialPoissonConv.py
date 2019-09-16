import pyro.distributions as dist

import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import \
    broadcast_all, probs_to_logits, lazy_property, logits_to_probs

from numbers import Number


class TorchNegativeBinomialPoissonConv(Distribution):
    r"""
    Creates a distribution for a random variable Z = X + Y such that:

        X ~ NegativeBinomial(:attr:`mu`, :attr:`alpha`)
        Y ~ Poisson(:attr:`lam`)

    where :attr:`mu` is the mean of X, `alpha` is the inverse of the overdispersion of X, and
    :attr:`lam` is the rate of Y.

    Args:
        mu (Number, Tensor): mean of the negative binomial variable
        alpha (Number, Tensor): inverse overdispersion of the negative binomial variable
        lam (Number, Tensor): rate of the Poisson variable
    """
    arg_constraints = {'mu': constraints.positive,
                       'alpha': constraints.positive,
                       'lam': constraints.positive}
    support = constraints.nonnegative_integer

    def __init__(self, mu, alpha, lam, max_poisson, validate_args=None):
        assert isinstance(max_poisson, int)
        self.mu, self.alpha, self.lam = broadcast_all(mu, alpha, lam)
        self.max_poisson = max_poisson
        self.neg_inf = torch.Tensor([float("-inf")]).to(device=mu.device)
        if isinstance(mu, Number) and isinstance(alpha, Number) \
                and isinstance(lam, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(TorchNegativeBinomialPoissonConv,
              self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TorchNegativeBinomialPoissonConv,
                                         _instance)
        batch_shape = torch.Size(batch_shape)
        new.mu = self.mu.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.lam = self.lam.expand(batch_shape)
        new.max_poisson = self.max_poisson
        new.neg_inf = self.neg_inf

        super(TorchNegativeBinomialPoissonConv,
              new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @staticmethod
    def _poisson_log_prob(lam, value):
        return (lam.log() * value) - lam - (value + 1).lgamma()

    @staticmethod
    def _neg_binom_log_prob(mu, alpha, value):
        return ((value + alpha).lgamma() - (value + 1).lgamma() - alpha.lgamma()
                + alpha * (alpha.log() - (alpha + mu).log())
                + value * (mu.log() - (alpha + mu).log()))

    @staticmethod
    def _poisson_log_prob_zero(lam):
        return - lam

    @staticmethod
    def _neg_binom_log_prob_zero(mu, alpha):
        return alpha * (alpha.log() - (alpha + mu).log())

    @staticmethod
    def _poisson_log_prob_one(lam):
        return lam.log() - lam

    @staticmethod
    def _neg_binom_log_prob_one(mu, alpha):
        # log_gamma(2.) = 0
        # log_gamma(z + 1) - log_gamma(z) = ln(z)
        return (alpha + 1) * (alpha.log() - (alpha + mu).log()) + mu.log()

    @staticmethod
    def _gamma_log_prob(alpha, beta, value):
        return (alpha * beta.log() +
                (alpha - 1) * value.log() -
                beta * value - alpha.lgamma())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        mu, alpha, lam, value = broadcast_all(self.mu, self.alpha, self.lam, value)

        # let us treat the assumed-to-be-prevalent edge case of value = 0 more efficiently
        zero_indices = (value == 0)
        poisson_log_prob_zero = self._poisson_log_prob_zero(lam[zero_indices])
        neg_binom_log_prob_zero = self._neg_binom_log_prob_zero(mu[zero_indices],
                                                                alpha[zero_indices])
        total_log_prob_zero = poisson_log_prob_zero + neg_binom_log_prob_zero

        # let us treat the assumed-to-be-prevalent edge case of value = 1 more efficiently
        one_indices = (value == 1)
        poisson_log_prob_zero = self._poisson_log_prob_zero(lam[one_indices])
        neg_binom_log_prob_zero = self._neg_binom_log_prob_zero(mu[one_indices],
                                                                alpha[one_indices])
        poisson_log_prob_one = self._poisson_log_prob_one(lam[one_indices])
        neg_binom_log_prob_one = self._neg_binom_log_prob_one(mu[one_indices],
                                                              alpha[one_indices])
        total_log_prob_one = torch.logsumexp(
            torch.cat(((poisson_log_prob_zero + neg_binom_log_prob_one).unsqueeze(-1),
                       (poisson_log_prob_one + neg_binom_log_prob_zero).unsqueeze(-1)), dim=-1),
            dim=-1, keepdim=False)

        # set up tables to perform complete numerical convolution on values > 1
        summation_indices = (value > 1)
        summation_obs = value[summation_indices]

        # estimate a reasonable low-end to begin the Poisson summation
        poisson_values_low = (lam.detach()[summation_indices] - self.max_poisson / 2).int()
        poisson_values_low = torch.clamp(torch.min(poisson_values_low,
                                                   (summation_obs
                                                    - self.max_poisson).int()), min=0)

        increment_values = torch.arange(
            0, self.max_poisson + 1, dtype=value.dtype,
            device=value.device).expand(summation_obs.shape + (-1,))
        poisson_values = poisson_values_low.unsqueeze(-1) + increment_values
        nb_values = summation_obs.unsqueeze(-1) - poisson_values

        # clamp nb_values to positive values
        inf_mask = (nb_values < 0)
        nb_values = torch.clamp(nb_values, min=0)

        # calculate poisson and negative binomial log probs on forward and backward counts
        poisson_log_prob_rest = self._poisson_log_prob(lam[summation_indices].unsqueeze(-1),
                                                       poisson_values.float())
        neg_binom_log_prob_rest = self._neg_binom_log_prob(mu[summation_indices].unsqueeze(-1),
                                                           alpha[summation_indices].unsqueeze(-1),
                                                           nb_values.float())
        total_log_prob_rest = poisson_log_prob_rest + neg_binom_log_prob_rest

        # set out-of-range values to -inf
        # total_log_prob_nnz[inf_mask] = float("-inf")  # this is doing a deep copy
        total_log_prob_rest = torch.where(inf_mask,
                                          self.neg_inf,
                                          total_log_prob_rest)

        total_log_prob_rest = torch.logsumexp(total_log_prob_rest,
                                              dim=-1,
                                              keepdim=False)

        out = torch.empty_like(value)
        out[zero_indices] = total_log_prob_zero  # deep copy
        out[one_indices] = total_log_prob_one  # deep copy
        out[summation_indices] = total_log_prob_rest  # deep copy

        return out


# We wrap the Torch distribution inside a Pyro distribution.
# This is as simple as inheriting
# distributions.torch_distribution.TorchDistributionMixin.
# It adds the required extra attributes.
class NegativeBinomialPoissonConv(TorchNegativeBinomialPoissonConv,
                                  dist.torch_distribution.TorchDistributionMixin):
    pass
