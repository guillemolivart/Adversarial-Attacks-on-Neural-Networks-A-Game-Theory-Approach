"""
Projected Gradient Descent (PGD) attack.

Mathematical definition
-----------------------
PGD (Madry et al., 2018) is a multi-step iterative attack that applies
projected gradient ascent on the loss within an :math:`\\ell_\\infty`
ball:

.. math::

    x^{(0)} &= x + u, \\quad u \\sim \\mathcal{U}(-\\varepsilon,\\,\\varepsilon)^d \\\\
    x^{(t+1)} &= \\Pi_{\\mathcal{B}_\\infty(x,\\,\\varepsilon)}\\!\\left(
                    x^{(t)} + \\alpha\\cdot
                    \\operatorname{sign}\\!\\left(
                        \\nabla_{x^{(t)}}\\,
                        \\ell\\bigl(f_\\theta(x^{(t)}),\\,y\\bigr)
                    \\right)
                  \\right),

where :math:`\\Pi_{\\mathcal{B}_\\infty(x,\\varepsilon)}` denotes the
projection onto the :math:`\\ell_\\infty` ball of radius
:math:`\\varepsilon` centred at the clean input :math:`x`, and
:math:`\\alpha` is the step size.

References
----------
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).
*Towards Deep Learning Models Resistant to Adversarial Attacks*.
ICLR 2018. https://arxiv.org/abs/1706.06083
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adversarial_game_theory.attacks.base import AdversarialAttack


class PGD(AdversarialAttack):
    """Projected Gradient Descent adversarial attack.

    Iteratively applies sign-gradient steps constrained to the
    :math:`\\ell_\\infty`-ball of radius :math:`\\varepsilon`.

    Parameters
    ----------
    model : nn.Module
        The target classifier :math:`f_\\theta`.
    epsilon : float
        Perturbation budget :math:`\\varepsilon`.
    alpha : float
        Per-step size :math:`\\alpha`.  A common heuristic is
        :math:`\\alpha = 2.5\\,\\varepsilon / T`.
    num_steps : int
        Number of gradient steps :math:`T`.
    random_start : bool
        If ``True`` (default), initialise the perturbation uniformly at
        random within the :math:`\\varepsilon`-ball, which helps avoid
        local optima.

    Examples
    --------
    >>> attack = PGD(model, epsilon=0.03, alpha=0.007, num_steps=10)
    >>> x_adv = attack.perturb(x_clean, labels)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float,
        alpha: float,
        num_steps: int,
        random_start: bool = True,
    ) -> None:
        super().__init__(model, epsilon)
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}.")
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate PGD adversarial examples.

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

        x_adv = x.clone().detach()
        if self.random_start:
            noise = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv + noise, 0.0, 1.0)

        for _ in range(self.num_steps):
            x_adv = x_adv.requires_grad_(True)
            loss = self._cross_entropy_loss(self.model(x_adv), y)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # Project onto the ℓ∞ ball centred at x
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0.0, 1.0)

        self.model.train(training)
        return x_adv.detach()
