"""
Abstract base class for adversarial attacks.

Game-theoretic framing
----------------------
An adversarial attack can be viewed as a two-player zero-sum game:

* **Defender** (the classifier) :math:`f_\\theta` seeks to minimise the
  expected loss :math:`\\mathbb{E}_{(x,y)}[\\ell(f_\\theta(x), y)]`.
* **Attacker** seeks to maximise the loss subject to a perturbation
  budget :math:`\\|\\delta\\|_p \\le \\varepsilon`.

Formally the attacker solves

.. math::

    \\delta^* = \\arg\\max_{\\|\\delta\\|_p \\le \\varepsilon}
                \\ell\\bigl(f_\\theta(x + \\delta),\\, y\\bigr).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AdversarialAttack(ABC):
    """Abstract base class for all adversarial attacks.

    Every concrete attack must implement :meth:`perturb`, which receives a
    batch of clean inputs and their true labels and returns the corresponding
    adversarial examples.

    Parameters
    ----------
    model : nn.Module
        The target (victim) classifier.
    epsilon : float
        Maximum perturbation budget :math:`\\varepsilon` in the chosen
        :math:`\\ell_p` norm.
    """

    def __init__(self, model: nn.Module, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self.model = model
        self.epsilon = epsilon

    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples.

        Parameters
        ----------
        x : torch.Tensor
            Clean input batch of shape ``(N, *)``, values in ``[0, 1]``.
        y : torch.Tensor
            Ground-truth integer labels of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Adversarial examples of the same shape as ``x``, clipped to
            ``[0, 1]``.
        """

    @staticmethod
    def _cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the mean cross-entropy loss (convenience helper).

        Parameters
        ----------
        logits : torch.Tensor
            Raw network outputs of shape ``(N, C)``.
        labels : torch.Tensor
            Integer class indices of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Scalar mean cross-entropy loss.
        """
        return nn.functional.cross_entropy(logits, labels)
