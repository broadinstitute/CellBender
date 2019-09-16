import torch
import torch.nn as nn

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


class EncodeZ(nn.Module):
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

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 input_transform: str = None):
        super(EncodeZ, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = input_transform

        # Set up the linear transformations used in fully-connected layers.
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.loc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.sig_out = nn.Linear(hidden_dims[-1], output_dim)

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Define the forward computation to go from gene expression to latent
        # representation.

        # Transform input.
        x = x.reshape(-1, self.input_dim)
        x = transform_input(x, self.transform)

        # Compute the hidden layers.
        hidden = self.softplus(self.linears[0](x))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        # Compute the outputs: loc is any real number, scale must be positive.
        loc = self.loc_out(hidden)
        scale = torch.exp(self.sig_out(hidden))

        return {'loc': loc.squeeze(), 'scale': scale.squeeze()}


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
                 prior_log_cell_counts: int,  # prior on counts per cell
                 input_transform: Optional[str] = None):
        super(EncodeNonZLatents, self).__init__()
        self.n_genes = n_genes
        self.z_dim = z_dim
        self.transform = input_transform
        self.output_dim = 3

        # Values related to logit cell probability
        self.INITIAL_WEIGHT_FOR_LOG_COUNTS = 2.
        self.P_OUTPUT_SCALE = 1.
        self.log_count_crossover = log_count_crossover
        self.prior_log_cell_counts = prior_log_cell_counts

        # Values related to epsilon
        self.EPS_OUTPUT_SCALE = 0.05  # slows down learning for epsilon
        self.EPS_OUTPUT_MEAN = 1.

        # Set up the linear transformations used in fully-connected layers.

        self.linears = nn.ModuleList([nn.Linear(3 + self.n_genes,
                                                hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output = nn.Linear(hidden_dims[-1], self.output_dim)

        # Adjust initialization conditions to start with a reasonable output.
        self._weight_init()

        # Set up the non-linear activations.
        self.nonlin = nn.Softplus()
        self.softplus = nn.Softplus()

        # Set up the initial biases.
        self.offset = None

        # Set up the initial scaling for values of x.
        self.x_scaling = None

        # Set up initial values for overlap normalization.
        self.overlap_mean = None
        self.overlap_std = None

    def _weight_init(self):
        """Initialize neural network weights"""

        # Initialize p to be a sigmoid function of UMI counts.
        for linear in self.linears:
            with torch.no_grad():
                linear.weight[0][0] = 1.
        with torch.no_grad():
            self.output.weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
            # Prevent a negative weight from starting something inverted.
            self.output.weight[1][0] = torch.abs(self.output.weight[1][0])
            self.output.weight[2][0] = torch.abs(self.output.weight[2][0])

    def _poisson_log_prob(self, lam, value):
        return (lam.log() * value) - lam - (value + 1).lgamma()

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

        # Calculate a similarity between expression and ambient.
        if chi_ambient is not None:
            overlap = self._poisson_log_prob(lam=counts * chi_ambient.detach().unsqueeze(0),
                                             value=x).sum(dim=-1, keepdim=True)
            if self.overlap_mean is None:
                self.overlap_mean = (overlap.max() + overlap.min()) / 2
                self.overlap_std = overlap.max() - overlap.min()
            overlap = (overlap - self.overlap_mean) / self.overlap_std * 5
        else:
            overlap = torch.zeros_like(counts)

        # Apply transformation to data.
        x = transform_input(x, self.transform)

        # Calculate a scale factor (first time through) to control the input variance.
        if self.x_scaling is None:
            n_std_est = 10
            num = int(self.n_genes * 0.4)
            std_estimates = torch.zeros([n_std_est])
            for i in range(n_std_est):
                idx = torch.randperm(x.nelement())
                std_estimates[i] = x.view(-1)[idx][:num].std().item()
            robust_std = std_estimates.median().item()
            self.x_scaling = (1. / robust_std) / 100.  # Get values on a level field

        # Form a new input by concatenation.
        # Compute the hidden layers and the output.
        x_in = torch.cat((log_sum,
                          log_nnz,
                          overlap,
                          x * self.x_scaling),
                         dim=-1)

        hidden = self.nonlin(self.linears[0](x_in))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.nonlin(self.linears[i](hidden))

        out = self.output(hidden).squeeze(-1)

        if self.offset is None:

            self.offset = dict()

            # Heuristic for initialization of logit_cell_probability.
            cells = (log_sum > self.log_count_crossover).squeeze()
            if (cells.sum() > 0) and ((~cells).sum() > 0):
                cell_median = out[cells, 0].median().item()
                empty_median = out[~cells, 0].median().item()
                self.offset['logit_p'] = empty_median + (cell_median - empty_median) * 9. / 10
            else:
                self.offset['logit_p'] = 0.

            # Heuristic for initialization of d.
            self.offset['d'] = out[cells, 1].median().item()

            # Heuristic for initialization of epsilon.
            self.offset['epsilon'] = out[cells, 2].mean().item()

        p_y_logit = ((out[:, 0] - self.offset['logit_p'])
                     * self.P_OUTPUT_SCALE).squeeze()
        epsilon = self.softplus((out[:, 2] - self.offset['epsilon']).squeeze()
                                * self.EPS_OUTPUT_SCALE + self.EPS_OUTPUT_MEAN)

        return {'p_y': p_y_logit,
                'd_loc': self.softplus(out[:, 1] - self.offset['d']
                                       + self.softplus(log_sum.squeeze()
                                                       - self.log_count_crossover)
                                       + self.log_count_crossover).squeeze(),
                'epsilon': epsilon}


def transform_input(x: torch.Tensor, transform: str) -> torch.Tensor:
    """Transform input to encoder.

    Args:
        x: Input torch.Tensor
        transform: Specifies which transformation to perform.  Must be one of
            ['log', 'normalize'].

    Returns:
        Transformed input as a torch.Tensor of the same type and shape as x.

    """

    if transform is None:
        return x

    elif transform == 'log':
        x = x.log1p()
        return x

    elif transform == 'normalize':
        x = x / x.sum(dim=-1, keepdim=True)
        return x

    else:
        raise NotImplementedError("Specified an input transform that is not "
                                  "supported.  Choose from 'log' or 'normalize'.")
