import torch
from cellbender.remove_background.vae.base import FullyConnectedNetwork


class Decoder(FullyConnectedNetwork):
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
        use_layernorm: True to use LayerNorm after each hidden layer is
            computed, before the activation
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

    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)
        self.input_dim = input_dim
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.network(z))
