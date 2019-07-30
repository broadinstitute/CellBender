from typing import Callable
import torch


class PosteriorImportanceSampler(object):
    def __init__(self,
                 proposal_generator: Callable[[int], torch.Tensor],
                 proposal_log_prob_function: Callable[[torch.Tensor], torch.Tensor],
                 prior_log_prob_function: Callable[[torch.Tensor], torch.Tensor],
                 model_log_like_function: Callable[[torch.Tensor], torch.Tensor],
                 log_objective_function: Callable[[torch.Tensor], torch.Tensor],
                 validate_callable_return_shapes: bool = True):
        """Initializer

        :param proposal_generator: a callable that generates proposals
        :param proposal_log_prob_function: a callable that returns the log likelihood of proposals
        :param prior_log_prob_function: a callable that returns the prior log probability on the proposals
        :param model_log_like_function: a callable the returns the molde log likelihood on the proposals
        :param log_objective_function: log of the objective function
        :param validate_callable_return_shapes: validate the returned tensor shape of callables

        .. note: ``proposal_generator`` takes an integer ``M`` (number of proposals) and returns a tensor
            with shape ``(M,) + batch_shape`` where ``batch_shape`` is the batch shape.

        .. note: ``proposal_log_prob_function``,  ``prior_log_prob_function``, ``model_log_like_function``, and
            ``log_objective_function`` each take a tensor with shape ``(M,) + batch_shape`` (proposals) and return
            a tensor with shape ``(M,) + batch_shape``.
        """

        self.proposal_generator = proposal_generator
        self.proposal_log_prob_function = proposal_log_prob_function
        self.prior_log_prob_function = prior_log_prob_function
        self.model_log_like_function = model_log_like_function
        self.log_objective_function = log_objective_function
        self.validate_callable_return_shapes = validate_callable_return_shapes

        self._batch_shape = None
        self._proposals_mb = None
        self._proposal_log_prob_mb = None
        self._prior_log_prob_mb = None
        self._model_log_like_mb = None
        self._log_obj_mb = None

    def run(self, n_particles: int, batch_shape: torch.Size) -> None:
        """Performs importance sampling and generates intermediate quantities

        :param n_particles: number of particles (proposals)
        :param batch_shape: batch shape
        """
        assert n_particles >= 1
        self._batch_shape = batch_shape
        proposal_shape = torch.Size((n_particles,) + self._batch_shape)

        proposals_mb = self.proposal_generator(n_particles)
        proposal_log_prob_mb = self.proposal_log_prob_function(self._proposals_mb)
        prior_log_prob_mb = self.prior_log_prob_function(self._proposals_mb)
        model_log_like_mb = self.model_log_like_function(self._proposals_mb)
        log_obj_mb = self.log_objective_function(self._proposals_mb)

        if self.validate_callable_return_shapes:
            assert proposals_mb.shape == proposal_shape
            assert proposal_log_prob_mb.shape == proposal_shape
            assert prior_log_prob_mb == proposal_shape
            assert model_log_like_mb.shape == proposal_shape
            assert log_obj_mb.shape == proposal_shape

        self._proposals_mb = proposals_mb
        self._proposal_log_prob_mb = proposal_log_prob_mb
        self._prior_log_prob_mb = prior_log_prob_mb
        self._model_log_like_mb = model_log_like_mb
        self._log_obj_mb = log_obj_mb

    @property
    def _intermediates_available(self):
        return ((self._batch_shape is not None) and
                (self._proposals_mb is not None) and
                (self._proposal_log_prob_mb is not None) and
                (self._prior_log_prob_mb is not None) and
                (self._model_log_like_mb is not None) and
                (self._log_obj_mb is not None))

    def ess(self) -> torch.Tensor:
        """Calculate the effective sample size (ESS)"""
        assert self._intermediates_available
        raise NotImplementedError

    @property
    def log_numerator(self) -> torch.Tensor:
        return torch.logsumexp(
            self._prior_log_prob_mb
            + self._model_log_like_mb
            + self._log_obj_mb
            - self._proposal_log_prob_mb, 0)

    @property
    def log_denominator(self) -> torch.Tensor:
        return torch.logsumexp(
            self._prior_log_prob_mb
            + self._model_log_like_mb
            - self._proposal_log_prob_mb, 0)

    @property
    def log_objective_posterior_expectation(self) -> torch.Tensor:
        """Calculates the log objective posterior expectation"""
        return self.log_numerator - self.log_denominator

