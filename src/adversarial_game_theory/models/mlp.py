"""
Multi-layer perceptron (fully-connected feed-forward network).

Mathematical definition
-----------------------
Given an input vector :math:`\\mathbf{x} \\in \\mathbb{R}^{d_0}`, an
*L*-layer MLP computes

.. math::

    \\mathbf{h}^{(0)} &= \\mathbf{x} \\\\
    \\mathbf{h}^{(\\ell)} &= \\sigma\\!\\left(W^{(\\ell)}\\,\\mathbf{h}^{(\\ell-1)}
                             + \\mathbf{b}^{(\\ell)}\\right),
                             \\quad \\ell = 1, \\dots, L-1 \\\\
    f(\\mathbf{x}) &= W^{(L)}\\,\\mathbf{h}^{(L-1)} + \\mathbf{b}^{(L)},

where :math:`\\sigma` is the activation function, :math:`W^{(\\ell)}`
and :math:`\\mathbf{b}^{(\\ell)}` are the weight matrix and bias of layer
:math:`\\ell`, and the final layer produces **logits** (no activation).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from adversarial_game_theory.models.base import BaseNetwork


class MLP(BaseNetwork):
    """Fully-connected multi-layer perceptron.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector :math:`d_0`.
    hidden_dims : Sequence[int]
        Width of each hidden layer :math:`(d_1, \\dots, d_{L-1})`.
    output_dim : int
        Number of output classes / logit dimensions :math:`d_L`.
    activation : nn.Module, optional
        Non-linearity applied after every hidden layer.
        Defaults to :class:`~torch.nn.ReLU`.
    dropout : float, optional
        Dropout probability applied after each hidden activation.
        Set to ``0.0`` (default) to disable dropout.

    Examples
    --------
    >>> model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    >>> x = torch.randn(32, 784)
    >>> logits = model(x)           # shape (32, 10)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(name="MLP")
        if activation is None:
            activation = nn.ReLU()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of flattened input vectors.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, input_dim)`` or any shape whose product
            equals ``input_dim`` (e.g. ``(batch_size, 1, 28, 28)`` for
            MNIST — the tensor is flattened automatically).

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch_size, output_dim)``.
        """
        return self.network(x.flatten(start_dim=1))
