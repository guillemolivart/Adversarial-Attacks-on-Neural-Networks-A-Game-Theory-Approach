"""Tests for game-theoretic implementations."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adversarial_game_theory.games.gan import GAN
from adversarial_game_theory.games.nash import NashEquilibriumSolver
from adversarial_game_theory.models.mlp import MLP


# ── Nash Equilibrium Tests ────────────────────────────────────────────────────

class TestNashEquilibriumSolver:
    def test_pure_strategy_game(self) -> None:
        """A game where one row dominates should assign all weight to that row."""
        # Attacker always prefers row 0; defender always prefers column 0.
        A = np.array([[2.0, -1.0], [-3.0, 1.0]])
        solver = NashEquilibriumSolver(A)
        eq = solver.solve()
        assert eq.attacker_strategy.shape == (2,)
        assert eq.defender_strategy.shape == (2,)
        assert abs(eq.attacker_strategy.sum() - 1.0) < 1e-6
        assert abs(eq.defender_strategy.sum() - 1.0) < 1e-6

    def test_symmetric_game(self) -> None:
        """Rock-Paper-Scissors has the uniform distribution as Nash equilibrium."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
        solver = NashEquilibriumSolver(A)
        eq = solver.solve()
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(eq.attacker_strategy, expected, atol=1e-4)
        np.testing.assert_allclose(eq.defender_strategy, expected, atol=1e-4)
        assert abs(eq.value) < 1e-4  # Zero-sum symmetric game has value 0

    def test_game_value_consistency(self) -> None:
        """v* = p*^T A q* should hold."""
        A = np.array([[3, -1], [-2, 4]], dtype=float)
        solver = NashEquilibriumSolver(A)
        eq = solver.solve()
        computed_value = eq.attacker_strategy @ A @ eq.defender_strategy
        assert abs(computed_value - eq.value) < 1e-4

    def test_strategies_are_probability_distributions(self) -> None:
        """Mixed strategies must be non-negative and sum to 1."""
        A = np.random.default_rng(42).uniform(-5, 5, size=(4, 3))
        solver = NashEquilibriumSolver(A)
        eq = solver.solve()
        assert np.all(eq.attacker_strategy >= -1e-8)
        assert np.all(eq.defender_strategy >= -1e-8)
        assert abs(eq.attacker_strategy.sum() - 1.0) < 1e-6
        assert abs(eq.defender_strategy.sum() - 1.0) < 1e-6

    def test_invalid_matrix_dimension(self) -> None:
        with pytest.raises(ValueError):
            NashEquilibriumSolver(np.array([1.0, 2.0, 3.0]))

    def test_single_strategy(self) -> None:
        """1×1 game: only one strategy available for each player."""
        A = np.array([[5.0]])
        solver = NashEquilibriumSolver(A)
        eq = solver.solve()
        assert abs(eq.value - 5.0) < 1e-4


# ── GAN Tests ─────────────────────────────────────────────────────────────────

LATENT_DIM = 8
IMAGE_DIM = 16


def _make_gan() -> GAN:
    """Build a tiny GAN for unit testing."""
    torch.manual_seed(0)
    generator = MLP(input_dim=LATENT_DIM, hidden_dims=[32], output_dim=IMAGE_DIM)
    discriminator = MLP(input_dim=IMAGE_DIM, hidden_dims=[32], output_dim=1)
    return GAN(generator, discriminator, latent_dim=LATENT_DIM)


def _make_loader(n: int = 64) -> DataLoader:
    torch.manual_seed(1)
    data = torch.rand(n, IMAGE_DIM)
    labels = torch.zeros(n, dtype=torch.long)
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


class TestGAN:
    def test_play_round_returns_floats(self) -> None:
        gan = _make_gan()
        loader = _make_loader()
        batch = next(iter(loader))[0]
        g_loss, d_loss = gan.play_round(batch)
        assert isinstance(g_loss, float)
        assert isinstance(d_loss, float)

    def test_train_returns_history(self) -> None:
        gan = _make_gan()
        loader = _make_loader()
        history = gan.train(num_rounds=4, dataloader=loader)
        assert len(history.attacker_losses) == 4
        assert len(history.defender_losses) == 4

    def test_generate_output_shape(self) -> None:
        gan = _make_gan()
        samples = gan.generate(num_samples=10)
        assert samples.shape == (10, IMAGE_DIM)

    def test_losses_are_finite(self) -> None:
        gan = _make_gan()
        loader = _make_loader()
        history = gan.train(num_rounds=4, dataloader=loader)
        assert all(np.isfinite(v) for v in history.attacker_losses)
        assert all(np.isfinite(v) for v in history.defender_losses)
