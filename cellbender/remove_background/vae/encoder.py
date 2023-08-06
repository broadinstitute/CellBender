import torch
import torch.nn as nn
import pyro
import numpy as np
from cellbender.remove_background.vae.base import FullyConnectedNetwork
from cellbender.remove_background import consts

from typing import Dict, List, Optional


class CompositeEncoder(dict):
    """A composite of several encoders to be run together on the same input.

    This represents an encoder that is a composite of several
    completely separate encoders.  The separate encoders are passed
    in as a dict, where keys are encoder names and values are encoder
    instances.  The output is another dict with the same keys, where values
    are the output tensors created by calling .forward(x) on those encoder
    instances.

    Attributes:
        module_dict: A dictionary of encoder modules.

    """

    def __init__(self, module_dict):
        super(CompositeEncoder, self).__init__(module_dict)
        self.module_dict = module_dict

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs) \
            -> Dict[str, torch.Tensor]:

        out = dict()
        # Encode z first.
        out['z'] = self.module_dict['z'].forward(**kwargs)

        # For each other module in the dict of the composite encoder,
        # call forward(), and pass in the encoded z.
        for key, value in self.module_dict.items():

            if key == 'z':
                continue  # already done

            out[key] = value.forward(**kwargs, z=out['z']['loc'].detach())

            if key == 'other':
                for subkey, value in out[key].items():
                    out[subkey] = value
                del out[key]

        return out


class EncodeZ(FullyConnectedNetwork):
    """Encoder module transforms gene expression into latent representation.

    The number of input units is the total number of genes and the number of
    output units is the dimension of the latent space.  This encoder transforms
    a point in high-dimensional gene expression space to a point in a
    low-dimensional latent space, via a learned transformation involving
    fully-connected layers.  The output is two vectors: one vector of latent
    variable means, and one vector of latent variable standard deviations.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of dimensions of the latent space in which gene
            expression will be embedded.
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        loc_out: torch.nn.Linear fully-connected output layer for the location
            of each point in latent space.
        sig_out: torch.nn.Linear fully-connected output layer for the standard
            deviation of each point in latent space.  Must result in positive
            values.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 input_transform: str = None,
                 **kwargs):
        assert len(hidden_dims) > 0, 'EncodeZ needs to have at least one hidden layer'
        super(EncodeZ, self).__init__(input_dim=input_dim,
                                      hidden_dims=hidden_dims[:-1],
                                      output_dim=hidden_dims[-1],
                                      hidden_activation=nn.Softplus(),
                                      output_activation=nn.Softplus(),
                                      norm_output=True,
                                      **kwargs)
        self.transform = input_transform
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.sig_out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        # Transform input.
        x = x.reshape(-1, self.input_dim)
        x_ = transform_input(x, self.transform)

        # Obtain last hidden layer.
        hidden = self.network(x_)

        # Compute the outputs: loc is any real number, scale must be positive.
        loc = self.loc_out(hidden)
        scale = torch.exp(self.sig_out(hidden))

        return {'loc': loc.squeeze(), 'scale': scale.squeeze()}


def _poisson_log_prob(lam, value):
    return (lam.log() * value) - lam - (value + 1).lgamma()


class EncodeNonZLatents(nn.Module):
    """Encoder module that transforms data into all latents except z.

    The number of input units is the total number of genes plus four
    hand-crafted features, and the number of output units is 3: these being
    latents logit_p, d, epsilon.  This encoder transforms
    a point in high-dimensional gene expression space into latents.  This
    encoder uses both the gene expression counts as well as an estimate of the
    ambient RNA profile in order to compute latents.

    Args:
        n_genes: Number of genes.  The size of the input of this encoder.
        z_dim: Dimension of latent encoding of gene expression, z.
        hidden_dims: Size of each of the hidden layers.
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center', None].
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.
        prior_log_cell_counts: Natural log of expected counts per cell.

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        output: torch.nn.Linear fully-connected output layer for the size
            of each input barcode.
        n_genes: Size of input gene expression.

    Returns:
        output: Dict containing -
            logit_p: Logit probability that droplet contains a cell
            d: Cell size scale factor
            epsilon: Value near one that represents droplet RT efficiency

    Notes:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].
        The output is in the form of a dict.  Ouput for cell probability is a
        logit, so can be any real number.  The transformation from logit to
        probability is a sigmoid.
        Several heuristics are used to try to encourage a good initialization.

    """

    def __init__(self,
                 n_genes: int,
                 z_dim: int,
                 log_count_crossover: float,  # prior on log counts of smallest cell
                 prior_log_cell_counts: float,  # prior on counts per cell
                 empty_log_count_threshold: float,
                 prior_logit_cell_prob: float,
                 input_transform: Optional[str] = None):
        super(EncodeNonZLatents, self).__init__()
        self.n_genes = n_genes
        self.z_dim = z_dim
        self.transform = input_transform
        self.output_dim = 1

        # Values related to logit cell probability
        self.INITIAL_WEIGHT_FOR_LOG_COUNTS = 1.
        self.P_OUTPUT_SCALE = 5.
        self.log_count_crossover = log_count_crossover
        self.empty_log_count_threshold = empty_log_count_threshold
        self.prior_log_cell_counts = prior_log_cell_counts
        self.prior_logit_cell_prob = prior_logit_cell_prob

        # Values related to epsilon
        self.EPS_OUTPUT_SCALE = 0.2  # slows down learning for epsilon
        self.EPS_OUTPUT_MEAN = 0.6931  # log(e - 1)  # softplus(0.6931) = 1.

        # Set up network for inference of p
        additional_features_p = 4
        self.layer1 = nn.Linear(additional_features_p + self.n_genes, 512)
        self.batchnorm1 = nn.BatchNorm1d(num_features=512)
        self.layer2 = nn.Linear(additional_features_p + 512, 512)
        self.batchnorm2 = nn.BatchNorm1d(num_features=512)
        self.layer3 = nn.Linear(additional_features_p + 512, 1)

        # Set up network for inference of epsilon
        additional_features_eps = 4
        self.eps_network = nn.Sequential(
            nn.Linear(additional_features_eps + self.n_genes, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Softplus(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Softplus(),
            nn.Linear(512, 1),
        )

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()
        self.dropout50 = nn.Dropout1d(p=0.5)

        # Set up the initial biases.
        self.offset = None

        # Set up the initial scaling for values of x.
        self.x_scaling = None
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.n_genes)

    def forward(self,
                x: torch.Tensor,
                chi_ambient: Optional[torch.Tensor],
                z: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        # Define the forward computation to go from gene expression to cell
        # probabilities.  The log of the total UMI counts is concatenated with
        # the input gene expression and the estimate of the difference between
        # the ambient RNA profile and this barcode's gene expression to form
        # an augmented input.

        x = x.reshape(-1, self.n_genes)

        # Calculate log total UMI counts per barcode.
        counts = x.sum(dim=-1, keepdim=True)
        log_sum = counts.log1p()

        # Calculate the log of the number of nonzero genes.
        log_nnz = (x > 0).sum(dim=-1, keepdim=True).float().log1p()

        # Calculate probability that log counts are consistent with d_empty.
        if chi_ambient is not None:
            # Gaussian log probability
            overlap = -0.5 * (
                    torch.clamp(log_sum - pyro.param("d_empty_loc").detach(), min=0.)
                    / 0.1
            ).pow(2)
        else:
            overlap = torch.zeros_like(counts)

        # Calculate a dot product between expression and ambient, for epsilon.
        if chi_ambient is not None:
            x_ambient = pyro.param("d_empty_loc").exp().detach() * chi_ambient.detach().unsqueeze(0)
            x_ambient_norm = x_ambient / torch.linalg.vector_norm(x_ambient, ord=2, dim=-1, keepdim=True)
            eps_overlap = (x_ambient_norm * x).sum(dim=-1, keepdim=True)
        else:
            eps_overlap = torch.zeros_like(counts)

        # Apply transformation to data.
        x = transform_input(x, self.transform)

        # Form a new input by concatenation.
        # Compute the hidden layers and the output.
        x_in = self.dropout50(self.batchnorm0(x))
        p_extra_features = torch.cat(
            (log_sum,
             log_nnz,
             overlap,
             torch.linalg.vector_norm(z.detach(), ord=2, dim=-1, keepdim=True)),
            dim=-1,
        )
        eps_extra_features = torch.cat(
            (log_sum,
             log_nnz,
             eps_overlap,
             torch.linalg.vector_norm(z.detach(), ord=2, dim=-1, keepdim=True)),
            dim=-1,
        )

        def add_extra_features(y, features):
            return torch.cat((features, y), dim=-1)

        # Do the forward pass for p
        x_ = self.softplus(self.batchnorm1(self.layer1(add_extra_features(x_in, p_extra_features))))
        x_ = self.softplus(self.batchnorm2(self.layer2(add_extra_features(x_, p_extra_features))))
        p_out = self.layer3(add_extra_features(x_, p_extra_features)).squeeze()

        # Do the forward pass for epsilon
        eps_out = self.eps_network(add_extra_features(x_in, eps_extra_features)).squeeze()

        if self.offset is None:

            self.offset = dict()

            # Heuristic for initialization of logit_cell_probability.
            cells = (log_sum > self.log_count_crossover).squeeze()
            assert cells.sum() > 4, "Fewer than 4 cells passed to encoder minibatch"
            self.offset['logit_p'] = p_out.mean().item()

            # Heuristic for initialization of epsilon.
            self.offset['epsilon'] = eps_out[cells].mean().item()

        p_y_logit = (
            (p_out - self.offset['logit_p'])
            + ((log_sum.squeeze() - self.log_count_crossover).abs().pow(0.5)
               * torch.sign(log_sum.squeeze() - self.log_count_crossover)
               * 10.)
        )

        # Enforce high cell prob for known cells
        beta = 50.  # like a temperature for the sigmoid's sharpness
        alpha = (beta * (log_sum - self.prior_log_cell_counts)).sigmoid().squeeze()
        p_y_logit = (1. - alpha) * p_y_logit + alpha * consts.REG_LOGIT_MEAN

        # Enforce low cell prob for known empties
        alpha_empty = (beta * (log_sum - self.empty_log_count_threshold)).sigmoid().squeeze()
        p_y_logit = alpha_empty * p_y_logit + (1. - alpha_empty) * (-1 * consts.REG_LOGIT_MEAN)

        # Constrain epsilon in (0.5, 2.5) with eps_out 0 mapping to epsilon 1
        # 1.0986122886681098 = log(3)
        epsilon = 2. * (eps_out * self.EPS_OUTPUT_SCALE - 1.0986122886681098).sigmoid() + 0.5

        d_empty = pyro.param("d_empty_loc").exp().detach()

        d_loc = self.softplus(
            (self.softplus(counts.squeeze() / (epsilon + 1e-2) - d_empty) + 1e-10).log()
            - self.log_count_crossover
        ) + self.log_count_crossover
        # d_loc = (d_loc + d_loc_est) / 2

        return {'p_y': p_y_logit,
                'd_loc': d_loc,
                'epsilon': epsilon}


def transform_input(x: torch.Tensor, transform: str, eps: float = 1e-5) -> torch.Tensor:
    """Transform input to encoder.

    Args:
        x: Input torch.Tensor
        transform: Specifies which transformation to perform.  Must be one of
            ['log', 'normalize', 'normalize_log', 'log_normalize'].
        eps: Preclude nan values in case of an input x with zero counts for a cell

    Returns:
        Transformed input as a torch.Tensor of the same type and shape as x.

    """

    if transform is None:
        return x

    elif transform == 'log':
        x = x.log1p()
        return x

    elif transform == 'normalize':
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x

    elif transform == 'normalize_log':
        x = x.log1p()
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        return x

    elif transform == 'log_normalize':
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x.log1p()
        return x

    else:
        raise NotImplementedError("Specified an input transform that is not "
                                  "supported.  Choose from 'log' or 'normalize'.")
