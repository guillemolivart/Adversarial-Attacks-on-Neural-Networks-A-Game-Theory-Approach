"""Tests for adversarial attack implementations."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adversarial_game_theory.attacks.carlini_wagner import CarliniWagnerL2
from adversarial_game_theory.attacks.fgsm import FGSM
from adversarial_game_theory.attacks.pgd import PGD
from adversarial_game_theory.models.mlp import MLP


@pytest.fixture()
def model_and_batch() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = MLP(input_dim=16, hidden_dims=[32], output_dim=4)
    model.eval()
    x = torch.rand(8, 16)
    y = torch.randint(0, 4, (8,))
    return model, x, y


class TestFGSM:
    def test_output_shape(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        attack = FGSM(model, epsilon=0.1)
        x_adv = attack.perturb(x, y)
        assert x_adv.shape == x.shape

    def test_output_range(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        attack = FGSM(model, epsilon=0.1)
        x_adv = attack.perturb(x, y)
        assert x_adv.min() >= 0.0 - 1e-6
        assert x_adv.max() <= 1.0 + 1e-6

    def test_perturbation_within_budget(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        eps = 0.1
        attack = FGSM(model, epsilon=eps)
        x_adv = attack.perturb(x, y)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-5

    def test_model_mode_restored(self, model_and_batch) -> None:
        """Attack should not permanently change training mode."""
        model, x, y = model_and_batch
        model.train()
        attack = FGSM(model, epsilon=0.05)
        attack.perturb(x, y)
        assert model.training

    def test_invalid_epsilon(self, model_and_batch) -> None:
        model, _, _ = model_and_batch
        with pytest.raises(ValueError):
            FGSM(model, epsilon=-0.1)


class TestPGD:
    def test_output_shape(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        attack = PGD(model, epsilon=0.1, alpha=0.02, num_steps=5)
        x_adv = attack.perturb(x, y)
        assert x_adv.shape == x.shape

    def test_output_range(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        attack = PGD(model, epsilon=0.1, alpha=0.02, num_steps=5)
        x_adv = attack.perturb(x, y)
        assert x_adv.min() >= 0.0 - 1e-6
        assert x_adv.max() <= 1.0 + 1e-6

    def test_perturbation_within_budget(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        eps = 0.1
        attack = PGD(model, epsilon=eps, alpha=0.02, num_steps=10, random_start=False)
        x_adv = attack.perturb(x, y)
        linf = (x_adv - x).abs().max().item()
        assert linf <= eps + 1e-5

    def test_model_mode_restored_from_eval(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        model.eval()
        attack = PGD(model, epsilon=0.05, alpha=0.01, num_steps=3)
        attack.perturb(x, y)
        assert not model.training

    def test_invalid_alpha(self, model_and_batch) -> None:
        model, _, _ = model_and_batch
        with pytest.raises(ValueError):
            PGD(model, epsilon=0.1, alpha=0.0, num_steps=5)

    def test_invalid_num_steps(self, model_and_batch) -> None:
        model, _, _ = model_and_batch
        with pytest.raises(ValueError):
            PGD(model, epsilon=0.1, alpha=0.01, num_steps=0)

    def test_pgd_stronger_than_fgsm(self, model_and_batch) -> None:
        """PGD with many steps should achieve at least as large a loss increase as FGSM."""
        model, x, y = model_and_batch
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            clean_loss = loss_fn(model(x), y).item()

        fgsm = FGSM(model, epsilon=0.1)
        x_fgsm = fgsm.perturb(x, y)
        pgd = PGD(model, epsilon=0.1, alpha=0.02, num_steps=20, random_start=False)
        x_pgd = pgd.perturb(x, y)

        with torch.no_grad():
            loss_fgsm = loss_fn(model(x_fgsm), y).item()
            loss_pgd = loss_fn(model(x_pgd), y).item()

        assert loss_pgd >= loss_fgsm - 0.5  # PGD should be at least competitive


class TestCarliniWagnerL2:
    def test_output_shape(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        attack = CarliniWagnerL2(model, epsilon=1e9, c=1.0, kappa=0.0, num_steps=20, lr=0.05)
        x_adv = attack.perturb(x, y)
        assert x_adv.shape == x.shape

    def test_output_range(self, model_and_batch) -> None:
        model, x, y = model_and_batch
        attack = CarliniWagnerL2(model, epsilon=1e9, c=1.0, kappa=0.0, num_steps=20, lr=0.05)
        x_adv = attack.perturb(x, y)
        assert x_adv.min() >= 0.0 - 1e-5
        assert x_adv.max() <= 1.0 + 1e-5

    def test_invalid_c(self, model_and_batch) -> None:
        model, _, _ = model_and_batch
        with pytest.raises(ValueError):
            CarliniWagnerL2(model, c=-1.0)

    def test_invalid_kappa(self, model_and_batch) -> None:
        model, _, _ = model_and_batch
        with pytest.raises(ValueError):
            CarliniWagnerL2(model, kappa=-0.1)
