import torch
import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    """Decoder module transforms latent representation into gene expression.

    The number of input units is the dimension of the latent space and the
    number of output units is the total number of genes.  This decoder
    transforms a point in low-dimensional latent space to a point in
    high-dimensional gene expression space, via a learned transformation
    involving fully-connected layers.  The output is a vector of fractional
    gene expression levels.  The output is on a simplex.  If the log_output
    attribute is False, the output thus sums to one, while setting log_output
    to True means that the sum of the exponential of each output sums to one.

    Args:
        input_dim: Number of dimensions of the latent space in which gene
            expression will be embedded.
        hidden_dims: Size of each of the hidden layers.
        output_dim: Number of genes.  The size of the output of this decoder.
        log_output: Whether or not the output is in log space.

    Attributes:
        linears: torch.nn.ModuleList of fully-connected layers before the
            output layer.
        outlinear: torch.nn.Linear fully-connected output layer.
        log_output: Whether or not the output is in log space.

    Note:
        A decoder with two hidden layers with sizes 100 and 500, respectively,
        should set hidden_dims = [100, 500].  A decoder with only one hidden
        layer should still pass in hidden_dims as a list, for example,
        hidden_dims = [500].

    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 log_output: bool = False):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.log_output = log_output

        # Set up the linear transformations used in fully-connected layers.
        self.linears = nn.ModuleList([nn.Linear(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):  # Second hidden layer onward
            self.linears.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.outlinear = nn.Linear(hidden_dims[-1], output_dim)

        # Set up the non-linear activations.
        self.softplus = nn.Softplus()
        if log_output:
            self.softmax = nn.LogSoftmax(dim=-1)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Define the forward computation to go from latent z to gene expression.

        # Compute the hidden layers.
        hidden = self.softplus(self.linears[0](z))
        for i in range(1, len(self.linears)):  # Second hidden layer onward
            hidden = self.softplus(self.linears[i](hidden))

        # Compute the output, which is on a simplex.
        gene_exp = self.softmax(self.outlinear(hidden))

        return gene_exp
