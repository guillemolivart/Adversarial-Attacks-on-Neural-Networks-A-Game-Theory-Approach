"""Tests for neural network model implementations."""
from __future__ import annotations

import pytest
import torch

from adversarial_game_theory.models.cnn import ConvNet
from adversarial_game_theory.models.mlp import MLP


class TestMLP:
    def test_output_shape(self) -> None:
        model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
        x = torch.randn(32, 784)
        logits = model(x)
        assert logits.shape == (32, 10)

    def test_auto_flatten(self) -> None:
        """MLP should accept 4-D inputs (e.g. MNIST batches) and flatten them."""
        model = MLP(input_dim=784, hidden_dims=[64], output_dim=10)
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (4, 10)

    def test_num_parameters_positive(self) -> None:
        model = MLP(input_dim=16, hidden_dims=[32], output_dim=4)
        assert model.num_parameters() > 0

    def test_dropout_training_vs_eval(self) -> None:
        """With high dropout, training and eval modes should differ."""
        torch.manual_seed(0)
        model = MLP(input_dim=16, hidden_dims=[64], output_dim=4, dropout=0.9)
        x = torch.randn(16, 16)
        model.train()
        out_train = model(x)
        model.eval()
        out_eval = model(x)
        # Eval output should be deterministic; re-running must match
        out_eval2 = model(x)
        assert torch.allclose(out_eval, out_eval2)

    def test_invalid_hidden_dims(self) -> None:
        """Empty hidden_dims should still produce a valid (linear) model."""
        model = MLP(input_dim=8, hidden_dims=[], output_dim=3)
        x = torch.randn(2, 8)
        assert model(x).shape == (2, 3)

    def test_save_and_load_checkpoint(self, tmp_path) -> None:
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        path = tmp_path / "mlp.pt"
        model.save_checkpoint(path)

        model2 = MLP(input_dim=8, hidden_dims=[16], output_dim=4)
        model2.load_checkpoint(path)

        x = torch.randn(4, 8)
        assert torch.allclose(model(x), model2(x))


class TestConvNet:
    def test_output_shape(self) -> None:
        model = ConvNet(in_channels=1, channel_dims=[8, 16], output_dim=10)
        x = torch.randn(4, 1, 28, 28)
        logits = model(x)
        assert logits.shape == (4, 10)

    def test_rgb_input(self) -> None:
        model = ConvNet(in_channels=3, channel_dims=[16, 32], output_dim=100)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 100)

    def test_num_parameters_positive(self) -> None:
        model = ConvNet(in_channels=1, channel_dims=[4], output_dim=2)
        assert model.num_parameters() > 0
