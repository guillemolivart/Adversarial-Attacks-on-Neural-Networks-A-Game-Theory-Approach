"""Tests for utility functions."""
from __future__ import annotations

import pytest
import torch

from adversarial_game_theory.models.mlp import MLP
from adversarial_game_theory.utils.metrics import (
    adversarial_accuracy,
    clean_accuracy,
    mean_perturbation_norm,
)
from adversarial_game_theory.utils.visualization import (
    plot_adversarial_examples,
    plot_training_history,
)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(99)
    model = MLP(input_dim=8, hidden_dims=[16], output_dim=3)
    model.eval()
    return model


class TestMetrics:
    def test_clean_accuracy_range(self, tiny_model) -> None:
        x = torch.rand(20, 8)
        y = torch.randint(0, 3, (20,))
        acc = clean_accuracy(tiny_model, x, y)
        assert 0.0 <= acc <= 1.0

    def test_adversarial_accuracy_range(self, tiny_model) -> None:
        x_adv = torch.rand(20, 8)
        y = torch.randint(0, 3, (20,))
        acc = adversarial_accuracy(tiny_model, x_adv, y)
        assert 0.0 <= acc <= 1.0

    def test_perfect_accuracy(self) -> None:
        """A model that always predicts class 0 should have 100% acc on y=0 data."""

        class AlwaysZero(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.size(0), 3)
                out[:, 0] = 1e6  # huge logit for class 0
                return out

        model = AlwaysZero()
        x = torch.rand(10, 3)
        y = torch.zeros(10, dtype=torch.long)
        assert clean_accuracy(model, x, y) == pytest.approx(1.0)

    def test_mean_perturbation_norm_zero_for_identical(self) -> None:
        x = torch.rand(8, 16)
        assert mean_perturbation_norm(x, x, p=2.0) == pytest.approx(0.0)

    def test_mean_perturbation_norm_l2(self) -> None:
        x = torch.zeros(4, 4)
        x_adv = torch.ones(4, 4)
        # Each sample has L2 norm = 2.0
        norm = mean_perturbation_norm(x, x_adv, p=2.0)
        assert norm == pytest.approx(2.0, abs=1e-5)

    def test_mean_perturbation_norm_linf(self) -> None:
        x = torch.zeros(4, 4)
        x_adv = torch.ones(4, 4) * 0.3
        norm = mean_perturbation_norm(x, x_adv, p=float("inf"))
        assert norm == pytest.approx(0.3, abs=1e-5)


class TestVisualization:
    def test_plot_adversarial_examples_returns_figure(self) -> None:
        x_clean = torch.rand(6, 1, 8, 8)
        x_adv = torch.rand(6, 1, 8, 8)
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_adversarial_examples(x_clean, x_adv, num_samples=3)
        assert fig is not None

    def test_plot_training_history_returns_figure(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_training_history([1.0, 0.9, 0.8], [0.5, 0.6, 0.7])
        assert fig is not None
