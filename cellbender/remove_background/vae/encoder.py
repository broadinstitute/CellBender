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


    # def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
    #              input_transform: str = None):
    #     super(EncodeZ, self).__init__()
    #     self.input_dim = input_dim
    #     self.output_dim = output_dim
    #     self.transform = input_transform
    #
    #     # Set up the linear transformations used in fully-connected layers.
    #     self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
    #     for i in range(1, len(hidden_dims)):  # Second hidden layer onward
    #         self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
    #     self.loc_out = nn.Linear(hidden_dims[-1], output_dim)
    #     self.sig_out = nn.Linear(hidden_dims[-1], output_dim)
    #
    #     # Set up the non-linear activations.
    #     self.softplus = nn.Softplus()
    #
    # def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
    #     # Define the forward computation to go from gene expression to latent
    #     # representation.
    #
    #     # Transform input.
    #     x = x.reshape(-1, self.input_dim)
    #     x = transform_input(x, self.transform)
    #
    #     # Compute the hidden layers.
    #     hidden = self.softplus(self.linears[0](x))
    #     for i in range(1, len(self.linears)):  # Second hidden layer onward
    #         hidden = self.softplus(self.linears[i](hidden))
    #
    #     # Compute the outputs: loc is any real number, scale must be positive.
    #     loc = self.loc_out(hidden)
    #     scale = torch.exp(self.sig_out(hidden))
    #
    #     return {'loc': loc.squeeze(), 'scale': scale.squeeze()}


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
                 hidden_dims: List[int],
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
        self.EPS_OUTPUT_SCALE = 1.  # slows down learning for epsilon
        self.EPS_OUTPUT_MEAN = 0.6931  # log(e - 1)  # 1.

        # Set up the linear transformations used in fully-connected layers.

        # self.linears = nn.ModuleList([nn.Linear(4 + self.n_genes,
        #                                         hidden_dims[0])])
        # for i in range(1, len(hidden_dims)):  # Second hidden layer onward
        #     self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        # self.output = nn.Linear(hidden_dims[-1], self.output_dim)

        n_extra_features = 4

        # Inject extra features at each level
        self.layer1 = nn.Linear(n_extra_features + self.n_genes, 32)
        self.batchnorm1 = nn.BatchNorm1d(num_features=32)
        self.layer2 = nn.Linear(n_extra_features + 32, 16)
        self.batchnorm2 = nn.BatchNorm1d(num_features=16)
        self.layer3 = nn.Linear(n_extra_features + 16, 3)

        # Adjust initialization conditions to start with a reasonable output.
        self._weight_init()

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

        # Set up the initial biases.
        self.offset = None

        # Set up the initial scaling for values of x.
        self.x_scaling = None
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.n_genes)

    def _weight_init(self):
        """Initialize neural network weights"""
        return

        # # Initialize p to be a sigmoid function of UMI counts.
        # for linear in self.linears:
        #     with torch.no_grad():
        #         linear.weight[0][0] = 1.
        # with torch.no_grad():
        #     self.output.weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS

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
        # if (counts < 1).sum() > 0:
        #     raise ValueError("A zero-count droplet has been passed to the encoder")
        log_sum = counts.log1p()

        # Calculate the log of the number of nonzero genes.
        log_nnz = (x > 0).sum(dim=-1, keepdim=True).float().log1p()

        # Calculate a similarity between expression and ambient.
        if chi_ambient is not None:
            overlap = _poisson_log_prob(
                lam=(pyro.param("d_empty_loc").exp().detach()
                     * chi_ambient.detach().unsqueeze(0) + 1e-10),
                value=x,
            ).sum(dim=-1, keepdim=True)

            # Normalize the values
            overlap_mean = overlap.mean()
            overlap_std = overlap.std() + 1e-3
            overlap = (overlap - overlap_mean) / overlap_std
        else:
            overlap = torch.zeros_like(counts)

        # Apply transformation to data.
        x = transform_input(x, self.transform)
        # print(f'x {x}')
        # print(f'there are {torch.isnan(x).sum()} nans in x')
        # print(f'there are {torch.isnan(x.sum(-1)).sum()} cells with nans in x')

        # # Calculate a scale factor (first time through) to control the input variance.
        # if self.x_scaling is None:
        #     x_cell = x[(counts > counts.median()).squeeze(), :]
        #     n_std_est = 10
        #     num = int(x_cell.nelement() / 2)
        #     std_estimates = torch.zeros([n_std_est])
        #     for i in range(n_std_est):
        #         idx = torch.randperm(x_cell.nelement())
        #         std_estimates[i] = x_cell.view(-1)[idx][:num].std().item()
        #     robust_std = torch.mean(std_estimates[~torch.isnan(std_estimates)]).item() + 1e-2
        #     self.x_scaling = (1. / robust_std) / 100.  # Get values on a level field

        # Form a new input by concatenation.
        # Compute the hidden layers and the output.
        x_in = self.batchnorm0(x)
        x_extra_features = torch.cat(
            (log_sum,
             log_nnz,
             overlap,
             torch.linalg.vector_norm(z.detach(), ord=2, dim=-1, keepdim=True)),
            dim=-1,
        )
        # print(f'x_in {x_in}')
        # print(f'there are {torch.isnan(x_in).sum()} nans in x_in')
        # print(f'there are {torch.isnan(x_in.sum(-1)).sum()} cells with nans in x_in')
        # ind = torch.isnan(x_in.sum(-1))
        # print(f'log_sum {log_sum[ind]}')
        # print(f'log_nnz {log_nnz[ind]}')
        # print(f'overlap {overlap[ind]}')
        # print(f'self.x_scaling {self.x_scaling}')

        # hidden = self.nonlin(self.linears[0](x_in))
        # for i in range(1, len(self.linears)):  # Second hidden layer onward
        #     hidden = self.nonlin(self.linears[i](hidden))
        #
        # out = self.output(hidden).squeeze(-1)

        def add_extra_features(y):
            return torch.cat((x_extra_features, y), dim=-1)

        # Do the forward pass
        x_ = self.softplus(self.batchnorm1(self.layer1(add_extra_features(x_in))))
        x_ = self.softplus(self.batchnorm2(self.layer2(add_extra_features(x_))))
        out = self.layer3(add_extra_features(x_))

        # Gather outputs
        p_out = out[:, 0].squeeze()
        eps_out = out[:, 1].squeeze()
        d_out = out[:, 2].squeeze()

        if self.offset is None:

            self.offset = dict()

            # Heuristic for initialization of logit_cell_probability.
            cells = (log_sum > self.log_count_crossover).squeeze()
            assert cells.sum() > 4, "Fewer than 4 cells passed to encoder minibatch"
            cell_prob = cells.sum() / x.shape[0]
            # self.offset['logit_p'] = torch.quantile(p_out, q=1. - cell_prob).item()
            self.offset['logit_p'] = p_out.mean().item()
            self.offset['logit_p_div'] = torch.quantile(p_out, q=0.95).item()

            # Heuristic for initialization of d.
            self.offset['d'] = d_out[cells].median().item()

            # Heuristic for initialization of epsilon.
            # self.offset['epsilon'] = out[cells, 2].mean().item()
            self.offset['epsilon'] = eps_out[cells].mean().item()

            # print(self.offset)

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

        epsilon = self.softplus((eps_out - self.offset['epsilon'])
                                * self.EPS_OUTPUT_SCALE + self.EPS_OUTPUT_MEAN)

        d_loc = self.softplus(
            d_out
            - self.offset['d']
            + self.softplus(log_sum.squeeze() - self.log_count_crossover)
            + self.log_count_crossover
        )

        return {'p_y': p_y_logit,
                'd_loc': d_loc,
                'epsilon': epsilon}

        # TODO: figure out why the list of droplets in the dataloader is different
        # TODO for the two test cases: with and without antibodies in file


# TODO: this was new
# class EncodeNonZLatents(nn.Module):
#     """Encoder module that transforms data into all latents except z.
#
#     The number of input units is the total number of genes plus four
#     hand-crafted features, and the number of output units is 3: these being
#     latents logit_p, d, epsilon.  This encoder transforms
#     a point in high-dimensional gene expression space into latents.  This
#     encoder uses both the gene expression counts as well as an estimate of the
#     ambient RNA profile in order to compute latents.
#
#     Args:
#         n_genes: Number of genes.  The size of the input of this encoder.
#         z_dim: Dimension of latent encoding of gene expression, z.
#         hidden_dims: Size of each of the hidden layers.
#         input_transform: Name of transformation to be applied to the input
#             gene expression counts.  Must be one of
#             ['log', 'normalize', 'log_center', None].
#         log_count_crossover: The log of the number of counts where the
#             transition from cells to empty droplets is expected to occur.
#         prior_log_cell_counts: Natural log of expected counts per cell.
#
#     Attributes:
#         transform: Name of transformation to be applied to the input gene
#             expression counts.
#         log_count_crossover: The log of the number of counts where the
#             transition from cells to empty droplets is expected to occur.
#         linears: torch.nn.ModuleList of fully-connected layers before the
#             output layer.
#         output: torch.nn.Linear fully-connected output layer for the size
#             of each input barcode.
#         n_genes: Size of input gene expression.
#
#     Returns:
#         output: Dict containing -
#             logit_p: Logit probability that droplet contains a cell
#             d: Cell size scale factor
#             epsilon: Value near one that represents droplet RT efficiency
#
#     Notes:
#         An encoder with two hidden layers with sizes 100 and 500, respectively,
#         should set hidden_dims = [100, 500].  An encoder with only one hidden
#         layer should still pass in hidden_dims as a list, for example,
#         hidden_dims = [500].
#         The output is in the form of a dict.  Ouput for cell probability is a
#         logit, so can be any real number.  The transformation from logit to
#         probability is a sigmoid.
#         Several heuristics are used to try to encourage a good initialization.
#
#     """
#
#     def __init__(self,
#                  n_genes: int,
#                  z_dim: int,
#                  hidden_dims: List[int],
#                  log_count_crossover: float,  # prior on log counts of smallest cell
#                  prior_log_cell_counts: int,  # prior on counts per cell
#                  input_transform: Optional[str] = None):
#         super(EncodeNonZLatents, self).__init__()
#         self.n_genes = n_genes
#         self.z_dim = z_dim
#         self.transform = input_transform
#         self.output_dim = 1
#
#         # Values related to logit cell probability
#         self.INITIAL_WEIGHT_FOR_LOG_COUNTS = 1.
#         self.P_OUTPUT_SCALE = 1.
#         self.log_count_crossover = log_count_crossover
#         self.prior_log_cell_counts = prior_log_cell_counts
#
#         # Values related to epsilon
#         self.EPS_OUTPUT_SCALE = 1.  # TODO: this doesn't work as advertised
#         self.EPS_OUTPUT_MEAN = 0.5413248  # correct for taking softplus of 1: ln(e^1 - 1)
#
#         # Set up the non-linear activations.
#         # self.nonlin = nn.Softplus()
#         self.softplus = nn.Softplus()
#
#         # Set up the linear transformations used in fully-connected layers.
#         self.count_embedding_layers = nn.ModuleList([nn.Linear(self.n_genes, hidden_dims[0])])
#         for i in range(1, len(hidden_dims)):  # Second hidden layer onward
#             self.count_embedding_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
#             self.count_embedding_layers.append(self.softplus)
#         self.count_embedding = nn.Sequential(*self.count_embedding_layers)
#
#         # Set up the linear readout for p.
#         self.p_output = nn.Linear(3 + self.z_dim + hidden_dims[-1], self.output_dim)
#
#         # Set up the linear readout for d.
#         self.d_output = nn.Linear(1 + self.z_dim, self.output_dim)
#
#         # Set up the linear readout for epsilon.
#         self.eps_output = nn.Linear(5 + self.z_dim + hidden_dims[-1], self.output_dim)
#
#         # Adjust initialization conditions to start with a reasonable output.
#         self._weight_init()
#
#         # Set up the initial biases.
#         self.offset_p = None
#         self.offset_d = None
#         self.offset_eps = None
#
#         # Set up the initial scaling for values of x.
#         self.x_scaling = None
#
#         # Set up initial values for overlap normalization.
#         self.overlap_mean = None
#         self.overlap_std = None
#
#     def _weight_init(self):
#         """Initialize neural network weights"""
#
#         with torch.no_grad():
#             # Initialize p to be a sigmoid function of UMI counts.
#             self.p_output.weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
#             # Initialize eps to be positively-correlated with UMI counts.
#             self.eps_output.weight[0][0] = -1 * self.INITIAL_WEIGHT_FOR_LOG_COUNTS
#
#     def _poisson_log_prob(self, lam, value):
#         return (lam.log() * value) - lam - (value + 1).lgamma()
#
#     def forward(self,
#                 x: torch.Tensor,
#                 chi_ambient: Optional[torch.Tensor],
#                 z: torch.Tensor,
#                 **kwargs) -> Dict[str, torch.Tensor]:
#         # Define the forward computation to go from gene expression to cell
#         # probabilities.  The log of the total UMI counts is concatenated with
#         # the input gene expression and the estimate of the difference between
#         # the ambient RNA profile and this barcode's gene expression to form
#         # an augmented input.
#
#         x = x.reshape(-1, self.n_genes)
#
#         # Calculate log total UMI counts per barcode.
#         counts = x.sum(dim=-1, keepdim=True)
#         log_sum = counts.log1p()
#
#         # Calculate the log of the number of nonzero genes.
#         log_nnz = (x > 0).sum(dim=-1, keepdim=True).float().log1p()
#
#         # Calculate a similarity between expression and ambient.
#         if chi_ambient is not None:
#             overlap = self._poisson_log_prob(lam=counts * chi_ambient.detach().unsqueeze(0),
#                                              value=x).sum(dim=-1, keepdim=True)
#             if self.overlap_mean is None:
#                 self.overlap_mean = (overlap.max() + overlap.min()) / 2
#                 self.overlap_std = overlap.max() - overlap.min()
#             overlap = (overlap - self.overlap_mean) / self.overlap_std * 5
#         else:
#             overlap = torch.zeros_like(counts)
#
#         # Apply transformation to data.
#         x = transform_input(x, self.transform)
#
#         # Calculate a scale factor (first time through) to control the input variance.
#         if self.x_scaling is None:
#             n_std_est = 10
#             num = int(self.n_genes * 0.4)
#             std_estimates = torch.zeros([n_std_est])
#             for i in range(n_std_est):
#                 idx = torch.randperm(x.nelement())
#                 std_estimates[i] = x.view(-1)[idx][:num].std().item()
#             robust_std = std_estimates.median().item()
#             self.x_scaling = (1. / robust_std) / 100.  # Get values on a level field
#
#         # Create an embedding of the counts.
#         embedding = self.count_embedding(x * self.x_scaling)
#
#         # Form a new input by concatenation and compute p.
#         x_p = torch.cat((log_sum, log_nnz, overlap, z, embedding), dim=-1)
#         p_out = self.p_output(x_p)
#
#         # Form a new input by concatenation and compute cell size factors.
#         # x_d = torch.cat((p_out, x_p), dim=-1)  # TODO: should I detach????
#         x_d = torch.cat((p_out.detach(), z.detach()), dim=-1)  # TODO: testing 9/21/20
#         d_out = self.d_output(x_d)
#
#         # Form a new input by concatenation and compute droplet efficiencies.
#         # x_eps = torch.cat((d_out, x_d), dim=-1)  # TODO: should I detach????
#         x_eps = torch.cat((d_out.detach(), p_out.detach(), x_p), dim=-1)  # TODO: testing 9/21/20
#         eps_out = self.eps_output(x_eps)
#
#         # Heuristic for initialization of outputs.
#         if self.offset_p is None:
#             cells = (log_sum > self.log_count_crossover).squeeze()
#             if (cells.sum() > 0) and ((~cells).sum() > 0):
#                 cell_median = p_out[cells].median().item()
#                 empty_median = p_out[~cells].median().item()
#                 self.offset_p = empty_median + (cell_median - empty_median) * 0.9
#             else:
#                 print('WARNING: Error in initialization of inference network for cell probabilities! '
#                       'Making a guess. There are too few cells per minibatch.')
#                 self.offset_p = p_out.median().item() + 3.
#             self.offset_d = d_out[cells].median().item()
#             self.offset_eps = eps_out[cells].median().item()
#
#         # Scale outputs appropriately, for good initialization.
#         p_y_logit = ((p_out - self.offset_p) * self.P_OUTPUT_SCALE).squeeze()
#         d_loc = self.softplus(d_out.squeeze() - self.offset_d
#                               + self.softplus(log_sum.squeeze() - self.log_count_crossover)
#                               + self.log_count_crossover).squeeze()
#         epsilon = self.softplus((eps_out - self.offset_eps).squeeze()
#                                 * self.EPS_OUTPUT_SCALE + self.EPS_OUTPUT_MEAN)
#
#         # TODO: gating epsilon so that it gets defined as 1 for empties
#         prob = p_y_logit.detach().sigmoid()  # Logits to probability
#         epsilon = (prob * epsilon + (1 - prob) * 1.)
#
#         # TODO: gating d so that it gets defined as cell prior for empties
#         d_loc = (prob * d_loc + (1 - prob) * kwargs['cell_prior_log'])
#
#         # TODO: try clipping outputs to safe ranges (to prevent nans / overflow)
#
#         # # TODO: testing eps = 1
#         # epsilon = torch.ones_like(epsilon)
#
#         return {'p_y': p_y_logit,
#                 'd_loc': d_loc,
#                 'epsilon': epsilon}


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
