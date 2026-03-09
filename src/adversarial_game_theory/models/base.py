"""
Abstract base class for neural network models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseNetwork(ABC, nn.Module):
    """Abstract base class for all neural network models in this project.

    Every concrete network must implement :meth:`forward`.  The class also
    provides convenience helpers for saving / loading checkpoints and for
    counting trainable parameters.

    Parameters
    ----------
    name : str
        Human-readable identifier used when saving checkpoints.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, *input_shape)``.

        Returns
        -------
        torch.Tensor
            Output logits of shape ``(batch_size, num_classes)``.
        """

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns
        -------
        int
            Sum of elements across all parameter tensors that require a
            gradient.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist model weights to disk.

        Parameters
        ----------
        path : str or Path
            Destination file path (usually with a ``.pt`` extension).
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path, device: torch.device | None = None) -> None:
        """Restore model weights from a checkpoint file.

        Parameters
        ----------
        path : str or Path
            Source file path previously created by :meth:`save_checkpoint`.
        device : torch.device, optional
            Target device for the loaded tensors.  Defaults to the current
            device of the model.
        """
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state)
