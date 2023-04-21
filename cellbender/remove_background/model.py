"""Definition of the model and the inference setup, with helper functions."""

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
import pyro.poutine as poutine

from cellbender.remove_background.vae import encoder as encoder_module
import cellbender.remove_background.consts as consts
from cellbender.remove_background.distributions.NegativeBinomialPoissonConv \
    import NegativeBinomialPoissonConv as NBPC
from cellbender.remove_background.distributions.NegativeBinomialPoissonConvApprox \
    import NegativeBinomialPoissonConvApprox as NBPCapprox
from cellbender.remove_background.distributions.NullDist import NullDist
from cellbender.remove_background.exceptions import NanException

from typing import Optional, Union, Dict, Callable
from numbers import Number


def calculate_lambda(model_type: str,
                     epsilon: torch.Tensor,
                     chi_ambient: torch.Tensor,
                     d_empty: torch.Tensor,
                     y: Union[torch.Tensor, None] = None,
                     d_cell: Union[torch.Tensor, None] = None,
                     rho: Union[torch.Tensor, None] = None,
                     chi_bar: Union[torch.Tensor, None] = None):
    """Calculate noise rate based on the model."""

    if model_type == "simple" or model_type == "ambient":
        lam = epsilon.unsqueeze(-1) * d_empty.unsqueeze(-1) * chi_ambient

    elif model_type == "swapping":
        lam = (rho.unsqueeze(-1) * chi_bar * epsilon.unsqueeze(-1) *
               (y.unsqueeze(-1) * d_cell.unsqueeze(-1) + d_empty.unsqueeze(-1)))

    elif model_type == "full":
        lam = (epsilon.unsqueeze(-1)
               * ((1. - rho.unsqueeze(-1)) * chi_ambient * d_empty.unsqueeze(-1)
                  + rho.unsqueeze(-1) * chi_bar * (y.unsqueeze(-1) * d_cell.unsqueeze(-1)
                                                   + d_empty.unsqueeze(-1))))
    else:
        raise NotImplementedError(f"model_type was set to {model_type}, "
                                  f"which is not implemented.")

    return lam


def calculate_mu(model_type: str,
                 epsilon: torch.Tensor,
                 d_cell: torch.Tensor,
                 chi: torch.Tensor,
                 y: Union[torch.Tensor, None] = None,
                 rho: Union[torch.Tensor, None] = None):
    """Calculate mean expression based on the model."""

    if model_type == 'simple':
        mu = epsilon.unsqueeze(-1) * d_cell.unsqueeze(-1) * chi

    elif model_type == 'ambient':
        mu = (y.unsqueeze(-1) * epsilon.unsqueeze(-1)
              * d_cell.unsqueeze(-1) * chi)

    elif model_type == 'swapping' or model_type == 'full':
        mu = ((1. - rho.unsqueeze(-1))
              * y.unsqueeze(-1) * epsilon.unsqueeze(-1)
              * d_cell.unsqueeze(-1) * chi)

    else:
        raise NotImplementedError(f"model_type was set to {model_type}, "
                                  f"which is not implemented.")

    return mu


class RemoveBackgroundPyroModel(nn.Module):
    """Class that contains the model and guide used for variational inference.

    Args:
        model_type: Which model is being used, one of ['simple', 'ambient',
            'swapping', 'full'].
        encoder: An instance of an encoder object.  Can be a CompositeEncoder.
        decoder: An instance of a decoder object.
        dataset_obj_priors: Dict which contains relevant priors.
        use_cuda: Will use GPU if True.
        analyzed_gene_names: Here only so that when we save a checkpoint, if we
            ever want to look at it, we will know which genes are which.
        phi_loc_prior: Mean of gamma distribution for global overdispersion.
        phi_scale_prior: Scale of gamma distribution for global overdispersion.
        rho_alpha_prior: Param of beta distribution for swapping fraction.
        rho_beta_prior: Param of beta distribution for swapping fraction.
        use_exact_log_prob: False (typical usage) to use an approximate log_prob
            computation, which is faster than an exact calculation.

    Attributes:
        All the above, plus
        device: Either 'cpu' or 'cuda' depending on value of use_cuda.
        loss: Dict that records information about the loss during training.

    """

    def __init__(self,
                 model_type: str,
                 encoder: Union[nn.Module, encoder_module.CompositeEncoder],
                 decoder: nn.Module,
                 dataset_obj_priors: Dict[str, float],
                 n_analyzed_genes: int,
                 n_droplets: int,
                 analyzed_gene_names: np.ndarray,
                 empty_UMI_threshold: int,
                 log_counts_crossover: float,
                 use_cuda: bool,
                 phi_loc_prior: float = consts.PHI_LOC_PRIOR,
                 phi_scale_prior: float = consts.PHI_SCALE_PRIOR,
                 rho_alpha_prior: float = consts.RHO_ALPHA_PRIOR,
                 rho_beta_prior: float = consts.RHO_BETA_PRIOR,
                 epsilon_prior: float = consts.EPSILON_PRIOR,
                 use_exact_log_prob: bool = consts.USE_EXACT_LOG_PROB):
        super(RemoveBackgroundPyroModel, self).__init__()

        self.model_type = model_type
        self.include_empties = True
        if self.model_type == "simple":
            self.include_empties = False
        self.include_rho = False
        if (self.model_type == "full") or (self.model_type == "swapping"):
            self.include_rho = True

        self.n_genes = n_analyzed_genes
        self.n_droplets = n_droplets
        self.analyzed_gene_names = analyzed_gene_names
        self.z_dim = decoder.input_dim
        self.encoder = encoder
        self.decoder = decoder
        self.use_exact_log_prob = use_exact_log_prob
        self.loss = {'train': {'epoch': [], 'elbo': []},
                     'test': {'epoch': [], 'elbo': []},
                     'learning_rate': {'epoch': [], 'value': []}}
        self.empty_UMI_threshold = empty_UMI_threshold
        self.counts_crossover = np.exp(log_counts_crossover)

        # Determine whether we are working on a GPU.
        if use_cuda:
            # Calling cuda() here will put all the parameters of
            # the encoder and decoder networks into GPU memory.
            self.cuda()
            try:
                for key, value in self.encoder.items():
                    value.cuda()
            except KeyError:
                pass
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.use_cuda = use_cuda

        # Priors
        assert dataset_obj_priors['d_std'] > 0, \
            f"Issue with prior: d_std is {dataset_obj_priors['d_std']}, " \
            f"but should be > 0."
        assert dataset_obj_priors['cell_counts'] > 0, \
            f"Issue with prior: cell_counts is " \
            f"{dataset_obj_priors['cell_counts']}, but should be > 0."

        self.d_cell_loc_prior = torch.tensor(np.log1p(dataset_obj_priors['cell_counts']))\
            .float().to(self.device)

        self.d_cell_scale_prior = torch.tensor(dataset_obj_priors['d_std']).to(self.device)
        self.z_loc_prior = torch.zeros(torch.Size([self.z_dim])).to(self.device)
        self.z_scale_prior = torch.ones(torch.Size([self.z_dim])).to(self.device)
        self.epsilon_prior = torch.tensor(epsilon_prior).to(self.device)

        self.phi_loc_prior = (phi_loc_prior
                              * torch.ones(torch.Size([])).to(self.device))
        self.phi_scale_prior = (phi_scale_prior
                                * torch.ones(torch.Size([])).to(self.device))
        self.phi_conc_prior = ((phi_loc_prior ** 2 / phi_scale_prior ** 2)
                               * torch.ones(torch.Size([])).to(self.device))
        self.phi_rate_prior = ((phi_loc_prior / phi_scale_prior ** 2)
                               * torch.ones(torch.Size([])).to(self.device))

        if self.model_type != "simple":

            assert dataset_obj_priors['empty_counts'] > 0, \
                f"Issue with prior: empty_counts should be > 0, but is " \
                f"{dataset_obj_priors['empty_counts']}"
            chi_ambient_sum = dataset_obj_priors['chi_ambient'].sum()
            assert np.isclose(a=chi_ambient_sum, b=[1.], atol=1e-5), \
                f"Issue with prior: chi_ambient should sum to 1, but it sums " \
                f"to {chi_ambient_sum}"
            chi_bar_sum = dataset_obj_priors['chi_bar'].sum()
            assert np.isclose(a=chi_bar_sum, b=[1.], atol=1e-5), \
                f"Issue with prior: chi_bar should sum to 1, but is {chi_bar_sum}"

            self.d_empty_loc_prior = (np.log1p(dataset_obj_priors['empty_counts'],
                                               dtype=np.float32).item()
                                      * torch.ones(torch.Size([]))
                                      .to(self.device))

            self.d_empty_scale_prior = (dataset_obj_priors['d_empty_std']
                                        * torch.ones(torch.Size([])).to(self.device))

            self.p_logit_prior = (dataset_obj_priors['cell_logit']
                                  * torch.ones(torch.Size([])).to(self.device))

            self.chi_ambient_init = dataset_obj_priors['chi_ambient'].to(self.device)
            self.avg_gene_expression = dataset_obj_priors['chi_bar'].to(self.device)

            self.empty_UMI_threshold = (torch.tensor(empty_UMI_threshold)
                                        .float().to(self.device))

        else:

            self.avg_gene_expression = None

        self.rho_alpha_prior = rho_alpha_prior * torch.ones(torch.Size([])).to(self.device)
        self.rho_beta_prior = rho_beta_prior * torch.ones(torch.Size([])).to(self.device)

    def _calculate_mu(self, **kwargs):
        return calculate_mu(model_type=self.model_type, **kwargs)

    def _calculate_lambda(self, **kwargs):
        return calculate_lambda(model_type=self.model_type, **kwargs)

    def model(self, x: torch.Tensor):
        """Data likelihood model.

            Args:
                x: Mini-batch of data. Barcodes are rows, genes are columns.

        """

        # Register the decoder with pyro.
        pyro.module("decoder", self.decoder, update_module_params=True)

        # Register the hyperparameter for ambient gene expression.
        if self.include_empties:
            chi_ambient = pyro.param("chi_ambient",
                                     self.chi_ambient_init *
                                     torch.ones(torch.Size([])).to(self.device),
                                     constraint=constraints.simplex)
        else:
            chi_ambient = None

        POISSON_APPROX = False

        if not POISSON_APPROX:
            # Sample phi from Gamma prior.
            phi = pyro.sample("phi",
                              dist.Gamma(self.phi_conc_prior,
                                         self.phi_rate_prior))

        # Happens in parallel for each data point (cell barcode) independently:
        with pyro.plate("data", x.shape[0],
                        use_cuda=self.use_cuda, device=self.device):

            # Sample z from prior.
            z = pyro.sample("z",
                            dist.Normal(loc=self.z_loc_prior,
                                        scale=self.z_scale_prior)
                            .expand_by([x.size(0)]).to_event(1))

            # Decode the latent code z to get fractional gene expression, chi.
            chi = pyro.deterministic("chi", self.decoder(z))

            # Sample d_cell based on priors.
            d_cell = pyro.sample("d_cell",
                                 dist.LogNormal(loc=self.d_cell_loc_prior,
                                                scale=self.d_cell_scale_prior)
                                 .expand_by([x.size(0)]))

            # Sample swapping fraction rho.
            if self.include_rho:
                rho = pyro.sample("rho", dist.Beta(self.rho_alpha_prior,
                                                   self.rho_beta_prior)
                                  .expand_by([x.size(0)]))
            else:
                rho = None

            # Sample epsilon based on priors.
            epsilon = pyro.sample("epsilon",
                                  dist.Gamma(concentration=self.epsilon_prior,
                                             rate=self.epsilon_prior)
                                  .expand_by([x.size(0)]))

            # If modelling empty droplets:
            if self.include_empties:

                # Sample d_empty based on priors.
                d_empty = pyro.sample("d_empty",
                                      dist.LogNormal(loc=self.d_empty_loc_prior,
                                                     scale=self.d_empty_scale_prior)
                                      .expand_by([x.size(0)]))

                # Sample y, the presence of a real cell, based on p_logit_prior.
                y = pyro.sample("y",
                                dist.Bernoulli(logits=self.p_logit_prior - 100.)  # TODO
                                .expand_by([x.size(0)]))

            else:
                d_empty = None
                y = None

            # Calculate the mean gene expression counts (for each barcode).
            mu_cell = self._calculate_mu(epsilon=epsilon,
                                         d_cell=d_cell,
                                         chi=chi,
                                         y=y,
                                         rho=rho)

            if self.include_empties:

                # Calculate the background rate parameter (for each barcode).
                lam = self._calculate_lambda(epsilon=epsilon,
                                             chi_ambient=chi_ambient,
                                             d_empty=d_empty,
                                             y=y,
                                             d_cell=d_cell,
                                             rho=rho,
                                             chi_bar=self.avg_gene_expression)
            else:
                lam = torch.zeros([self.n_genes]).to(self.device)

            if POISSON_APPROX:
                # Data distributed as the sum of two Poissons.
                c = pyro.sample("obs",
                                dist.Poisson(rate=mu_cell + lam + consts.POISSON_EPS_SAFEGAURD).to_event(1),
                                obs=x.reshape(-1, self.n_genes))
                alpha = None

            else:
                alpha = phi.reciprocal()

                if not consts.USE_EXACT_LOG_PROB:

                    # Use a negative binomial approximation as the observation model.
                    c = pyro.sample("obs", NBPCapprox(mu=mu_cell + consts.NBPC_MU_EPS_SAFEGAURD,
                                                      alpha=alpha + consts.NBPC_ALPHA_EPS_SAFEGAURD,
                                                      lam=lam + consts.NBPC_LAM_EPS_SAFEGAURD).to_event(1),
                                    obs=x.reshape(-1, self.n_genes))

                else:
                    c = pyro.sample("obs", NBPC(mu=mu_cell + consts.NBPC_MU_EPS_SAFEGAURD,
                                                alpha=alpha + consts.NBPC_ALPHA_EPS_SAFEGAURD,
                                                lam=lam + consts.NBPC_LAM_EPS_SAFEGAURD,
                                                max_poisson=100).to_event(1),
                                    obs=x.reshape(-1, self.n_genes))

            if self.include_empties:

                # Put a prior on p_y_logit to maintain balance.
                pyro.sample("p_logit_reg", dist.Normal(loc=self.p_logit_prior,
                                                       scale=(consts.P_LOGIT_SCALE
                                                              * torch.ones([1]).to(self.device))))

                # Additionally use the surely empty droplets for regularization,
                # since we know these droplets by their UMI counts.
                counts = x.sum(dim=-1, keepdim=False)
                surely_empty_mask = (counts < self.empty_UMI_threshold).bool().to(self.device)
                surely_cell_mask = (counts >= self.d_cell_loc_prior.exp()).bool().to(self.device)

                with poutine.mask(mask=surely_empty_mask):

                    with poutine.scale(scale=consts.REG_SCALE_AMBIENT_EXPRESSION):

                        if self.include_rho:
                            r = rho.detach()
                        else:
                            r = None

                        # Semi-supervision of ambient expression using all empties.
                        lam = self._calculate_lambda(epsilon=epsilon.detach(),
                                                     chi_ambient=chi_ambient,
                                                     d_empty=d_empty,
                                                     y=torch.zeros_like(d_empty),
                                                     d_cell=d_cell.detach(),
                                                     rho=r,
                                                     chi_bar=self.avg_gene_expression)
                        pyro.sample("obs_empty",
                                    dist.Poisson(rate=lam + consts.POISSON_EPS_SAFEGAURD).to_event(1),
                                    obs=x.reshape(-1, self.n_genes))

                # Grab our posterior for the logit cell probability (this is a workaround).
                p_logit_posterior = pyro.sample("p_passback",
                                                NullDist(torch.zeros(1).to(self.device))
                                                .expand_by([x.size(0)]))

                empty_constraint = soft_constraint_function(loc=-2, scale=2, lower_bound=False)
                cell_constraint = soft_constraint_function(loc=2, scale=2, lower_bound=True)

                # For empties where our inference is currently wrong, impose a big penalty.
                with poutine.mask(mask=surely_empty_mask):

                    # Supervision of empty probabilities.
                    pyro.factor("obs_surely_empty_y",
                                log_factor=empty_constraint(p_logit_posterior))

                # For cells where our inference is currently wrong, impose a big penalty.
                with poutine.mask(mask=surely_cell_mask):

                    # Supervision of cell probabilities.
                    pyro.factor("obs_surely_cell_y",
                                log_factor=cell_constraint(p_logit_posterior))

                # Softer semi-supervision to encourage cell probabilities to do the right thing.
                probably_empty_mask = (counts < self.counts_crossover).bool().to(self.device)
                probably_cell_mask = (counts >= self.counts_crossover).bool().to(self.device)

                with poutine.mask(mask=probably_empty_mask):
                    with poutine.scale(scale=consts.REG_SCALE_SOFT_SUPERVISION):
                        pyro.sample("obs_probably_empty_y",
                                    dist.Normal(loc=-1 * torch.ones_like(y) * consts.REG_LOGIT_MEAN,
                                                scale=consts.REG_LOGIT_SOFT_SCALE),
                                    obs=p_logit_posterior)

                with poutine.mask(mask=probably_cell_mask):
                    with poutine.scale(scale=consts.REG_SCALE_SOFT_SUPERVISION):
                        pyro.sample("obs_probably_cell_y",
                                    dist.Normal(loc=torch.ones_like(y) * consts.REG_LOGIT_MEAN,
                                                scale=consts.REG_LOGIT_SOFT_SCALE),
                                    obs=p_logit_posterior)

        # Regularization of epsilon.mean()
        if surely_cell_mask.sum() >= 2:
            epsilon_mean = epsilon[surely_cell_mask].mean()
            with poutine.scale(scale=surely_cell_mask.sum() / 10.):
                pyro.sample("epsilon_mean",
                            dist.Normal(loc=epsilon_mean, scale=0.01),
                            obs=torch.ones_like(epsilon_mean))

        return {'chi_ambient': chi_ambient, 'z': z,
                'mu': mu_cell, 'lam': lam, 'alpha': alpha, 'counts': c}

    @config_enumerate(default='parallel')
    def guide(self, x: torch.Tensor):
        """Variational posterior.

            Args:
                x: Mini-batch of data. Barcodes are rows, genes are columns.

        """

        nan_check = False

        if nan_check:
            for param in pyro.get_param_store().keys():
                if torch.isnan(pyro.param(param).sum()):
                    raise NanException(param)

        # Register the encoder(s) with pyro.
        for name, module in self.encoder.items():
            pyro.module("encoder_" + name, module, update_module_params=True)

        # Initialize variational parameters for d_cell.
        d_cell_scale = pyro.param("d_cell_scale",
                                  torch.tensor([consts.D_CELL_SCALE_INIT]).to(self.device),
                                  constraint=constraints.positive)

        if self.include_empties:

            # Initialize variational parameters for d_empty.
            d_empty_loc = pyro.param("d_empty_loc",
                                     self.d_empty_loc_prior *
                                     torch.ones(torch.Size([])).to(self.device),
                                     constraint=constraints.positive)
            d_empty_scale = pyro.param("d_empty_scale",
                                       self.d_empty_scale_prior *
                                       torch.ones(torch.Size([]))
                                       .to(self.device),
                                       constraint=constraints.positive)

            # Register the hyperparameter for ambient gene expression.
            chi_ambient = pyro.param("chi_ambient",
                                     self.chi_ambient_init *
                                     torch.ones(torch.Size([])).to(self.device),
                                     constraint=constraints.simplex)

        # Initialize variational parameters for rho.
        if self.include_rho:
            rho_alpha = pyro.param("rho_alpha",
                                   self.rho_alpha_prior *
                                   torch.ones(torch.Size([])).to(self.device),
                                   constraint=constraints.interval(consts.RHO_PARAM_MIN,
                                                                   consts.RHO_PARAM_MAX))  # Prevent NaNs
            rho_beta = pyro.param("rho_beta",
                                  self.rho_beta_prior *
                                  torch.ones(torch.Size([])).to(self.device),
                                  constraint=constraints.interval(consts.RHO_PARAM_MIN,
                                                                  consts.RHO_PARAM_MAX))

        # Initialize variational parameters for phi.
        phi_loc = pyro.param("phi_loc",
                             self.phi_loc_prior *
                             torch.ones(torch.Size([])).to(self.device),
                             constraint=constraints.positive)
        phi_scale = pyro.param("phi_scale",
                               self.phi_scale_prior *
                               torch.ones(torch.Size([])).to(self.device),
                               constraint=constraints.positive)

        # Sample phi from a Gamma distribution (after re-parameterization).
        phi_conc = phi_loc.pow(2) / phi_scale.pow(2)
        phi_rate = phi_loc / phi_scale.pow(2)
        pyro.sample("phi", dist.Gamma(phi_conc, phi_rate))

        # Happens in parallel for each data point (cell barcode) independently:
        with pyro.plate("data", x.shape[0],
                        use_cuda=self.use_cuda, device=self.device):

            # Sample swapping fraction rho.
            if self.include_rho:
                rho = pyro.sample("rho", dist.Beta(rho_alpha,
                                                   rho_beta).expand_by([x.size(0)]))

            # Encode the latent variables from the input gene expression counts.
            if self.include_empties:

                # Sample d_empty, which doesn't depend on y.
                d_empty = pyro.sample("d_empty",
                                      dist.LogNormal(loc=d_empty_loc,
                                                     scale=d_empty_scale)
                                      .expand_by([x.size(0)]))

                enc = self.encoder(x=x,
                                   chi_ambient=chi_ambient.detach(),
                                   cell_prior_log=self.d_cell_loc_prior)

            else:
                enc = self.encoder(x=x,
                                   chi_ambient=None,
                                   cell_prior_log=self.d_cell_loc_prior)

            # Code specific to models with empty droplets.
            if self.include_empties:

                # Regularize based on wanting a balanced p_y_logit.
                pyro.sample("p_logit_reg", dist.Normal(loc=enc['p_y'], scale=consts.P_LOGIT_SCALE))

                # Pass back the inferred p_y to the model.
                pyro.sample("p_passback", NullDist(enc['p_y'].detach()))

                # Sample the Bernoulli y from encoded p(y).
                y = pyro.sample("y", dist.Bernoulli(logits=enc['p_y']))

                # Gate d_cell_loc so empty droplets do not give big gradients.
                prob = enc['p_y'].sigmoid()  # Logits to probability
                d_cell_loc_gated = (prob * enc['d_loc'] + (1 - prob)
                                    * self.d_cell_loc_prior)  # NOTE: necessary to pass on sim6

                # Sample d based on the encoding.
                d_cell = pyro.sample("d_cell", dist.LogNormal(loc=d_cell_loc_gated,
                                                              scale=d_cell_scale))

                # Mask out empty droplets.
                with poutine.mask(mask=y.bool()):

                    # Sample latent code z for the barcodes containing cells.
                    z = pyro.sample("z", dist.Normal(loc=enc['z']['loc'], scale=enc['z']['scale'])
                                    .to_event(1))

                    # Gate epsilon and sample.
                    epsilon_gated = (prob * enc['epsilon'] + (1 - prob) * 1.)

                    epsilon = pyro.sample("epsilon", dist.Gamma(epsilon_gated * self.epsilon_prior,
                                                                self.epsilon_prior))

            else:

                # Sample d based on the encoding.
                pyro.sample("d_cell", dist.LogNormal(loc=enc['d_loc'], scale=d_cell_scale))

                # Sample latent code z for each cell.
                pyro.sample("z", dist.Normal(loc=enc['z']['loc'], scale=enc['z']['scale'])
                            .to_event(1))


def soft_constraint_function(loc: float, scale: float, lower_bound: bool, invert: bool = True) \
        -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function which implements an exponential soft constraint:
    exp((1 - 2 * lower_bound) * (x - loc) * scale)

    NOTE: if lower_bound, then if x > loc, this function returns 0, and if
    lower_bound = False, then if x < loc, this function returns 0

    Args:
        loc: value where constraint turns on
        scale: how fast the constraint blows up. 1 is exponential blow up.
        lower_bound: True to turn on constraint when x < loc, False to turn on
            constraint when x > loc
        invert: True to emit negative values as penalties

    Returns:
        Function that maps a value to a log prob factor that represents a
        soft constraint
    """
    sign = 1 - 2 * lower_bound
    inversion_sign = -1 if invert else 1

    def _fun(x: torch.Tensor) -> torch.Tensor:
        return inversion_sign * (sign * (x - loc) * scale).exp()
        # return inversion_sign * torch.clamp((sign * (x - loc) * scale).exp() - 1., min=0.)

    return _fun


def get_rho() -> Optional[np.ndarray]:
    """Get ambient RNA expression for 'empty' droplets.

    Return:
        chi_ambient: The ambient gene expression profile, as a normalized
            vector that sums to one.

    Note:
        Inference must have been performed on a model with a 'chi_ambient'
        hyperparameter prior to making this call.

    """

    rho = None

    if 'rho_alpha' in pyro.get_param_store().keys() \
            and 'rho_beta' in pyro.get_param_store().keys():
        rho = np.array([to_ndarray(pyro.param('rho_alpha')).item(),
                        to_ndarray(pyro.param('rho_beta')).item()])

    return rho


def get_param_store_key(key: str) -> Union[np.ndarray, None]:
    val = None

    if key in pyro.get_param_store().keys():
        val = to_ndarray(pyro.param(key)).squeeze()

    return val


def to_ndarray(x: Union[Number, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert a numeric value or array to a numpy array on cpu."""

    if type(x) is np.ndarray:
        return x

    elif type(x) is torch.Tensor:
        return x.detach().cpu().numpy()

    elif type(x) is Number:
        return np.array(x)

    else:
        raise TypeError(f'to_ndarray() received input of type {type(x)}')
