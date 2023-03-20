"""Base neural network architectures, for convenience"""

import torch
from typing import Optional, List


class Exp(torch.nn.Module):
    """Exponential activation function as a torch module"""

    def __init__(self, eps: float = 1e-5):
        """Exponential activation function with numerical stabilization, useful
        for outputs that must be > 0

        NOTE: output = torch.exp(input) + eps

        Parameters
        ----------
        eps: Numerical stability additive constant.
        """
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x) + self.eps

    def __repr__(self):
        return f"torch.exp() + {self.eps}"


class FullyConnectedLayer(torch.nn.Module):
    """Neural network unit made of a fully connected linear layer, but
    customizable including shapes, activations, batch norm, layer norm, and
    dropout.

    Parameters
    ----------
    input_dim: Number of features for input
    output_dim: Number of features for output
    activation: Activation function to be applied to each hidden layer
        (default :py:class:`torch.nn.ReLU`)
    use_batch_norm: True to apply batch normalization using
        :py:class:`torch.nn.BatchNorm1d` with ``momentum=0.01``, ``eps=0.001``
        (default False)
    use_layer_norm: True to apply layer normalization (after optional batch
        normalization) using :py:class:`torch.nn.LayerNorm` with
        ``elementwise_affine=False`` (default False)
    dropout_rate: Dropout rate to use in :py:class:`torch.nn.Dropout` before
        linear layer

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # set up layers as a list of Linear modules with appropriate extras
        modules = []
        if dropout_rate is not None:
            modules.append(torch.nn.Dropout(p=dropout_rate))
        modules.append(torch.nn.Linear(in_features=input_dim, out_features=output_dim))
        if use_batch_norm:
            modules.append(
                torch.nn.BatchNorm1d(num_features=output_dim, momentum=0.01, eps=0.001)
            )
        if use_layer_norm:
            modules.append(
                torch.nn.LayerNorm(
                    normalized_shape=output_dim, elementwise_affine=False
                )
            )
        if activation is not None:
            modules.append(activation)

        # concatenate Linear layers using Sequential
        self.layer = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FullyConnectedNetwork(torch.nn.Module):
    """Neural network made of fully connected linear layers,
    :py:class:`FullyConnectedLayer`. Architecture is customizable including
    shapes, activations, batch norm, layer norm, and dropout.

    Parameters
    ----------
    input_dim: Number of features for input
    hidden_dims: List of hidden layer sizes, can be empty list []
    output_dim: Number of features for output
    hidden_activation: Activation function to be applied to each hidden layer
        (default :py:class:`torch.nn.ReLU`)
    output_activation: Activation function to be applied to output (default None)
    use_batch_norm: True to apply batch normalization using
        :py:class:`torch.nn.BatchNorm1d` with ``momentum=0.01``, ``eps=0.001``
        (default False)
    use_layer_norm: True to apply layer normalization (after optional batch
        normalization) using :py:class:`torch.nn.LayerNorm` with
        ``elementwise_affine=False`` (default False)
    norm_output: True to apply normalization to output layer before output
        activation (default False)
    dropout_rate: Dropout rate to use in :py:class:`torch.nn.Dropout` for each
        hidden layer (applied before each layer)
    dropout_input: True to apply dropout before first layer (default False)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
        output_activation: Optional[torch.nn.Module] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        norm_output: bool = False,
        dropout_rate: Optional[float] = None,
        dropout_input: bool = False,
    ):
        super().__init__()

        if use_layer_norm and use_batch_norm:
            raise UserWarning(
                "You are trying to use both batch norm and layer "
                "norm. That's probably too much norm."
            )

        # set up layers as a list of Linear modules with appropriate extras
        dim_ins_and_outs = zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        n_layers = 1 + len(hidden_dims)
        layers = [
            FullyConnectedLayer(
                input_dim=i,
                output_dim=j,
                activation=hidden_activation
                if (layer < n_layers - 1)
                else output_activation,
                use_batch_norm=use_batch_norm
                if ((layer < n_layers - 1) or norm_output)
                else False,
                use_layer_norm=use_layer_norm
                if ((layer < n_layers - 1) or norm_output)
                else False,
                dropout_rate=None
                if ((layer == 0) and not dropout_input)
                else dropout_rate,
            )
            for layer, (i, j) in enumerate(dim_ins_and_outs)
        ]

        # concatenate Linear layers using Sequential
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
