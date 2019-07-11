import torch
import torch.nn as nn
from typing import Dict, List, Union
import warnings


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
        # For each module in the dict of the composite encoder, call forward().
        out = dict()
        for key, value in self.module_dict.items():
            out[key] = value.forward(**kwargs)

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


class EncodeD(nn.Module):
    """Encoder module that transforms gene expression into latent cell size.

    The number of input units is the total number of genes and the number of
    output units is the dimension of the latent space.  This encoder transforms
    a point in high-dimensional gene expression space to a latent cell size
    estimate, via a learned transformation involving fully-connected
    layers.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of dimensions of the size estimate (1).
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        param: The log of the number of counts where the transition from
            cells to empty droplets is expected to occur.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        output: torch.nn.Linear fully-connected output layer for the size
            of each input barcode.
        input_dim: Size of input gene expression.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].
        This encoder acts as if each barcode input contains a cell.  In reality,
        barcodes that do not contain a cell will not propagate gradients back
        to this encoder, due to the design of the rest of the model.

    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 input_transform: str = None, log_count_crossover: float = 7.):
        super(EncodeD, self).__init__()
        self.input_dim = input_dim
        self.transform = input_transform
        self.param = log_count_crossover

        # Set up the linear transformations used in fully-connected layers.
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.output = nn.Linear(hidden_dims[-1], output_dim)

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Define the forward computation to go from gene expression to cell
        # probabilities.

        # Transform input and calculate log total UMI counts per barcode.
        x = x.reshape(-1, self.input_dim)
        log_sum = x.sum(dim=-1, keepdim=True).log1p()
        x = transform_input(x, self.transform)

        # Compute the hidden layers and the output.
        hidden = self.softplus(self.linears[0](x))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        return self.softplus(self.output(hidden)
                             + self.softplus(log_sum - self.param)
                             + self.param).squeeze()


class EncodeNonEmptyDropletLogitProb(nn.Module):
    """Encoder module that transforms gene expression into cell probabilities.

    The number of input units is the total number of genes and the number of
    output units is the dimension of the latent space.  This encoder transforms
    a point in high-dimensional gene expression space to a latent probability
    that a given barcode contains a real cell, via a learned transformation
    involving fully-connected layers.  This encoder uses both the gene
    expression counts as well as an estimate of the ambient RNA profile in
    order to output a cell probability.

    Args:
        input_dim: Number of genes.  The size of the input of this encoder.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of dimensions of the probability estimate (1).
        input_transform: Name of transformation to be applied to the input
            gene expression counts.  Must be one of
            ['log', 'normalize', 'log_center'].
        log_count_crossover: The log of the number of counts where the
            transition from cells to empty droplets is expected to occur.

    Attributes:
        transform: Name of transformation to be applied to the input gene
            expression counts.
        param: The log of the number of counts where the transition from
            cells to empty droplets is expected to occur.
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        output: torch.nn.Linear fully-connected output layer for the size
            of each input barcode.
        input_dim: Size of input gene expression.

    Note:
        An encoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  An encoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].
        The output is in the form of a logit, so can be any real number.  The
        transformation from logit to probability is a sigmoid.

    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 log_count_crossover: float, input_transform: str = None):
        super(EncodeNonEmptyDropletLogitProb, self).__init__()
        self.input_dim = input_dim
        self.transform = input_transform
        self.param = log_count_crossover
        self.INITIAL_WEIGHT_FOR_LOG_COUNTS = 1.
        self.INITIAL_OUTPUT_SCALE_FOR_LOG_COUNTS = 5.
        self.INITIAL_OUTPUT_BIAS_FOR_LOG_COUNTS = \
            -1 * self.INITIAL_OUTPUT_SCALE_FOR_LOG_COUNTS

        # Set up the linear transformations used in fully-connected layers.
        # Adjust initialization conditions to start with a reasonable output.
        self.linears = nn.ModuleList([nn.Linear(1 + 2*input_dim,
                                                hidden_dims[0])])
        with torch.no_grad():
            self.linears[-1].weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            # Initialize p so that it starts out based (almost) on UMI counts.
            with torch.no_grad():
                self.linears[-1].weight[0][0] = self.INITIAL_WEIGHT_FOR_LOG_COUNTS
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        # Initialize p to be a sigmoid function of UMI counts.
        with torch.no_grad():
            self.output.weight[0][0] = self.INITIAL_OUTPUT_SCALE_FOR_LOG_COUNTS
            self.output.bias.data.copy_(torch.tensor([self.INITIAL_OUTPUT_BIAS_FOR_LOG_COUNTS
                                                      * log_count_crossover]))

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()

    def forward(self,
                x: torch.Tensor,
                chi_ambient: torch.Tensor) -> torch.Tensor:
        # Define the forward computation to go from gene expression to cell
        # probabilities.  The log of the total UMI counts is concatenated with
        # the input gene expression and the estimate of the difference between
        # the ambient RNA profile and this barcode's gene expression to form
        # an augmented input.

        # Transform input and calculate log total UMI counts per barcode.
        x = x.reshape(-1, self.input_dim)
        log_sum = x.sum(dim=-1, keepdim=True).log1p()
        x = transform_input(x, self.transform)

        # Form a new input by concatenation.
        # Compute the hidden layers and the output.
        hidden = self.softplus(self.linears[0](torch.cat((log_sum,
                                                          x,
                                                          x - chi_ambient),
                                                         dim=-1)))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        return self.output(hidden).squeeze(-1)


def transform_input(x: torch.Tensor, transform: str) -> Union[torch.Tensor,
                                                              None]:
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
        warnings.warn("Specified an input transform that is not "
                      "supported.  Choose from 'log' or 'normalize'.")
        return None
