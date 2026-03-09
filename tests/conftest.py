"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adversarial_game_theory.models.mlp import MLP


@pytest.fixture()
def simple_model() -> nn.Module:
    """A tiny pre-trained MLP for use in attack tests."""
    model = MLP(input_dim=16, hidden_dims=[32], output_dim=4)
    model.eval()
    return model


@pytest.fixture()
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A small batch of random 'images' and their labels."""
    torch.manual_seed(42)
    x = torch.rand(8, 16)
    y = torch.randint(0, 4, (8,))
    return x, y
