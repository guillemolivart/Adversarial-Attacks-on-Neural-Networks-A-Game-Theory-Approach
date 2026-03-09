"""
Convolutional neural network (CNN) for image classification.

Mathematical definition
-----------------------
A convolutional layer with kernel :math:`W \\in \\mathbb{R}^{C_{out}
\\times C_{in} \\times k \\times k}` computes

.. math::

    (W * x)[c, i, j] = \\sum_{c'} \\sum_{p,q} W[c, c', p, q]\\,
                        x\\!\\left[c',\\, i+p,\\, j+q\\right],

followed by batch normalisation and an activation :math:`\\sigma`.
Global average pooling collapses spatial dimensions before a linear
classifier head.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from adversarial_game_theory.models.base import BaseNetwork


class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d → BatchNorm → ReLU.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    kernel_size : int
        Spatial size of the convolutional kernel (assumed square).
    stride : int
        Convolution stride.
    padding : int
        Zero-padding added to both sides of the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Conv → BN → ReLU to ``x``."""
        return self.block(x)


class ConvNet(BaseNetwork):
    """Configurable convolutional neural network for image classification.

    Architecture
    ------------
    A stack of :class:`ConvBlock` modules (each followed by 2×2 max-pooling)
    is used as the feature extractor.  The spatial features are collapsed
    with global average pooling and mapped to class logits by a linear head.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (e.g. ``1`` for grayscale, ``3`` for
        RGB).
    channel_dims : Sequence[int]
        Number of output channels for each convolutional block.
    output_dim : int
        Number of output classes.
    dropout : float, optional
        Dropout probability applied before the final linear layer.
        Defaults to ``0.0`` (disabled).

    Examples
    --------
    >>> model = ConvNet(in_channels=1, channel_dims=[32, 64], output_dim=10)
    >>> x = torch.randn(8, 1, 28, 28)
    >>> logits = model(x)       # shape (8, 10)
    """

    def __init__(
        self,
        in_channels: int,
        channel_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(name="ConvNet")
        blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out in channel_dims:
            blocks.append(ConvBlock(c_in, c_out))
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            c_in = c_out
        self.feature_extractor = nn.Sequential(*blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(c_in, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Image batch of shape ``(batch_size, in_channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch_size, output_dim)``.
        """
        features = self.feature_extractor(x)
        pooled = self.global_avg_pool(features)
        flat = pooled.flatten(start_dim=1)
        return self.classifier(flat)
