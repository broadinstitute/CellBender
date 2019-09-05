from typing import Callable, NamedTuple
import torch
from pyro.distributions import TorchDistribution


class PosteriorImportanceSamplerInputs(NamedTuple):
    """Inputs for ``PosteriorImportanceSampler``

    :param proposal_distribution: proposal distribution
    :param prior_distribution: prior distribution
    :param log_likelihood_function: a callable the returns the model log likelihood on the proposals
    :param log_objective_function: log of the objective function

    .. note:: ``proposal_distribution`` and ``prior_distribution`` must have the same shape; both must have
        empty ``event_shape`` and the same ``batch_shape``.

    .. note:: ``log_like_function`` is a callable that takes the proposal tensor with shape
        ``(n_particles,) + batch_shape`` and returns a tensor with the same shape. Here, ``n_particles``
        is the number of proposals per batch dimension.

    .. note:: ``log_objective_function`` takes a tensor with shape ``(n_particles,) + batch_shape``
        (proposals) and return a tensor with shape `(n_outputs, n_particles) + batch_shape``, where
        ``n_outputs``. If the user wishes to calculate the posterior expectation of just a single
        quantity, then ``n_outputs`` must be set to ``1``.
    """
    proposal_distribution: TorchDistribution
    prior_distribution: TorchDistribution
    log_likelihood_function: Callable[[torch.Tensor], torch.Tensor]
    log_objective_function: Callable[[torch.Tensor], torch.Tensor]


class PosteriorImportanceSampler(object):
    def __init__(self,
                 inputs: PosteriorImportanceSamplerInputs,
                 validate_callable_return_shapes: bool = True):
        """Initializer
        :param inputs: as instance of ```PosteriorImportanceSamplerInputs``
        :param validate_callable_return_shapes: validate the returned tensor shape of callables
        """

        self.proposal_distribution = inputs.proposal_distribution
        self.prior_distribution = inputs.prior_distribution
        self.log_likelihood_function = inputs.log_likelihood_function
        self.log_objective_function = inputs.log_objective_function
        self.validate_callable_return_shapes = validate_callable_return_shapes

        self._batch_shape = None
        self._proposal_shape = None
        self._obj_shape = None
        self._proposals_mb = None
        self._proposal_log_prob_mb = None
        self._prior_log_prob_mb = None
        self._log_like_mb = None
        self._log_obj_kmb = None

    def run(self, n_particles: int, n_outputs: int, batch_shape: torch.Size) -> None:
        """Performs importance sampling and generates intermediate quantities

        :param n_particles: number of particles (proposals)
        :param n_outputs: number of outputs in the objective function (i.e. the leftmost dimension ``K``)
        :param batch_shape: batch shape
        """
        assert n_particles >= 1
        self._batch_shape = batch_shape
        self._proposal_shape = torch.Size((n_particles,) + self._batch_shape)
        self._obj_shape = torch.Size((n_outputs,) + self._proposal_shape)

        proposals_mb = self.proposal_distribution.sample(torch.Size((n_particles,)))
        proposal_log_prob_mb = self.proposal_distribution.log_prob(proposals_mb)
        prior_log_prob_mb = self.prior_distribution.log_prob(proposals_mb)
        model_log_like_mb = self.log_likelihood_function(proposals_mb)
        log_obj_kmb = self.log_objective_function(proposals_mb)

        if self.validate_callable_return_shapes:
            assert proposals_mb.shape == self._proposal_shape
            assert proposal_log_prob_mb.shape == self._proposal_shape
            assert prior_log_prob_mb.shape == self._proposal_shape
            assert model_log_like_mb.shape == self._proposal_shape
            assert log_obj_kmb.shape == self._obj_shape

        self._proposals_mb = proposals_mb
        self._proposal_log_prob_mb = proposal_log_prob_mb
        self._prior_log_prob_mb = prior_log_prob_mb
        self._log_like_mb = model_log_like_mb
        self._log_obj_kmb = log_obj_kmb

    @property
    def _intermediates_available(self):
        return ((self._batch_shape is not None) and
                (self._proposals_mb is not None) and
                (self._proposal_log_prob_mb is not None) and
                (self._prior_log_prob_mb is not None) and
                (self._log_like_mb is not None) and
                (self._log_obj_kmb is not None))

    @property
    def ess(self) -> torch.Tensor:
        """Calculate the effective sample size (ESS)"""
        assert self._intermediates_available
        log_w_kmb = (self._prior_log_prob_mb
                     + self._log_like_mb
                     + self._log_obj_kmb
                     - self._proposal_log_prob_mb)
        log_ess_kb = 2 * torch.logsumexp(log_w_kmb, 1) - torch.logsumexp(2 * log_w_kmb, 1)
        return log_ess_kb.exp()

    @property
    def log_numerator(self) -> torch.Tensor:
        return torch.logsumexp(
            self._prior_log_prob_mb
            + self._log_like_mb
            + self._log_obj_kmb
            - self._proposal_log_prob_mb, 1)

    @property
    def log_denominator(self) -> torch.Tensor:
        return torch.logsumexp(
            self._prior_log_prob_mb
            + self._log_like_mb
            - self._proposal_log_prob_mb, 0)

    @property
    def log_objective_posterior_expectation(self) -> torch.Tensor:
        """Calculates the log objective posterior expectation"""
        return self.log_numerator - self.log_denominator
