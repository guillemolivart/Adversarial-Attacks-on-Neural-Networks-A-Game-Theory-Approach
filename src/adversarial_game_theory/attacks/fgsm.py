"""
Fast Gradient Sign Method (FGSM).

Mathematical definition
-----------------------
Given a differentiable loss function :math:`\\ell`, a classifier
:math:`f_\\theta`, an input :math:`x \\in [0,1]^d` and its true label
:math:`y`, FGSM (Goodfellow et al., 2015) computes a single-step
:math:`\\ell_\\infty`-bounded perturbation:

.. math::

    \\delta^* = \\varepsilon \\cdot
                \\operatorname{sign}\\!\\left(
                    \\nabla_{x}\\,\\ell\\bigl(f_\\theta(x),\\, y\\bigr)
                \\right),

and returns the adversarial example

.. math::

    x^{\\mathrm{adv}} = \\operatorname{clip}_{[0,1]}(x + \\delta^*).

References
----------
Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015).
*Explaining and Harnessing Adversarial Examples*.
ICLR 2015. https://arxiv.org/abs/1412.6572
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adversarial_game_theory.attacks.base import AdversarialAttack


class FGSM(AdversarialAttack):
    """Fast Gradient Sign Method attack.

    Produces adversarial examples constrained to the
    :math:`\\ell_\\infty`-ball of radius :math:`\\varepsilon` around the
    original input.

    Parameters
    ----------
    model : nn.Module
        The target classifier :math:`f_\\theta`.
    epsilon : float
        Perturbation budget :math:`\\varepsilon \\in (0, 1]`.

    Examples
    --------
    >>> attack = FGSM(model, epsilon=0.03)
    >>> x_adv = attack.perturb(x_clean, labels)
    """

    def __init__(self, model: nn.Module, epsilon: float) -> None:
        super().__init__(model, epsilon)

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate FGSM adversarial examples.

        The model is temporarily set to evaluation mode and gradients are
        computed with respect to the input.

        Parameters
        ----------
        x : torch.Tensor
            Clean images of shape ``(N, *)``, values in ``[0, 1]``.
        y : torch.Tensor
            Ground-truth labels of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Adversarial images of shape ``(N, *)``, values in ``[0, 1]``.
        """
        training = self.model.training
        self.model.eval()

        x_adv = x.clone().detach().requires_grad_(True)
        loss = self._cross_entropy_loss(self.model(x_adv), y)
        loss.backward()

        with torch.no_grad():
            perturbation = self.epsilon * x_adv.grad.sign()
            x_adv = torch.clamp(x + perturbation, 0.0, 1.0)

        self.model.train(training)
        return x_adv.detach()
