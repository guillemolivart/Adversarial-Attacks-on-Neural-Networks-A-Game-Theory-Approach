"""
Carlini & Wagner (C&W) :math:`\\ell_2` attack.

Mathematical definition
-----------------------
C&W (Carlini & Wagner, 2017) reformulates the adversarial attack as an
optimisation problem that minimises the :math:`\\ell_2` distortion while
ensuring misclassification via an auxiliary objective:

.. math::

    \\min_{\\mathbf{w}}\\;
        \\|\\tfrac{1}{2}(\\tanh(\\mathbf{w})+1) - x\\|_2^2
        + c \\cdot f\\!\\left(\\tfrac{1}{2}(\\tanh(\\mathbf{w})+1)\\right)

where the change of variables :math:`x^{\\mathrm{adv}} =
\\tfrac{1}{2}(\\tanh(\\mathbf{w})+1)` automatically enforces
:math:`x^{\\mathrm{adv}} \\in [0,1]^d`, and the attack objective is

.. math::

    f(x') = \\max\\!\\Bigl(
                \\max_{j \\ne t}\\{Z(x')_j\\} - Z(x')_t,\\;
                -\\kappa
            \\Bigr),

with :math:`Z(x')` the pre-softmax logits, :math:`t` the target class
(the true class for an untargeted attack), and :math:`\\kappa \\ge 0` a
confidence margin.  The constant :math:`c > 0` trades distortion against
attack success.

References
----------
Carlini, N., & Wagner, D. (2017).
*Towards Evaluating the Robustness of Neural Networks.*
IEEE S&P 2017. https://arxiv.org/abs/1608.04644
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from adversarial_game_theory.attacks.base import AdversarialAttack


class CarliniWagnerL2(AdversarialAttack):
    """Carlini & Wagner :math:`\\ell_2` adversarial attack.

    Parameters
    ----------
    model : nn.Module
        The target classifier :math:`f_\\theta`.
    epsilon : float
        Upper bound on the :math:`\\ell_2` distortion.  Used only to
        filter examples that exceed the budget after optimisation.
        Set to a large value (e.g. ``1e9``) to keep all adversarial
        examples regardless of distortion.
    c : float
        Trade-off constant :math:`c > 0` between distortion and attack
        success.  Larger values prioritise misclassification.
    kappa : float
        Confidence margin :math:`\\kappa \\ge 0`.  Larger values make
        the attack find *more confident* misclassifications.
    num_steps : int
        Number of Adam optimisation steps.
    lr : float
        Learning rate for the Adam optimiser.

    Examples
    --------
    >>> attack = CarliniWagnerL2(model, epsilon=2.0, c=1.0, kappa=0.0,
    ...                          num_steps=100, lr=0.01)
    >>> x_adv = attack.perturb(x_clean, labels)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e9,
        c: float = 1.0,
        kappa: float = 0.0,
        num_steps: int = 100,
        lr: float = 0.01,
    ) -> None:
        super().__init__(model, epsilon)
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}.")
        if kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {kappa}.")
        self.c = c
        self.kappa = kappa
        self.num_steps = num_steps
        self.lr = lr

    def _attack_objective(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the C&W attack objective :math:`f(x')`.

        Parameters
        ----------
        logits : torch.Tensor
            Pre-softmax network output of shape ``(N, C)``.
        labels : torch.Tensor
            True class indices of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Per-sample objective values of shape ``(N,)``.
        """
        n = logits.size(0)
        true_logits = logits[torch.arange(n), labels]

        # Mask the true class to find max over other classes
        masked = logits.clone()
        masked[torch.arange(n), labels] = -float("inf")
        max_other = masked.max(dim=1).values

        return torch.clamp(max_other - true_logits, min=-self.kappa)

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate C&W :math:`\\ell_2` adversarial examples.

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
            Samples whose :math:`\\ell_2` distortion exceeds
            :attr:`epsilon` are returned unchanged.
        """
        training = self.model.training
        self.model.eval()

        # Change of variables: x_adv = 0.5 * (tanh(w) + 1)
        # Initialise w such that tanh(w) ≈ 2x - 1
        w = torch.atanh(torch.clamp(2.0 * x - 1.0, -1 + 1e-6, 1 - 1e-6)).detach().clone()
        w.requires_grad_(True)

        optimizer = optim.Adam([w], lr=self.lr)

        x_best = x.clone()
        best_l2 = torch.full((x.size(0),), float("inf"), device=x.device)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            x_adv = 0.5 * (torch.tanh(w) + 1.0)

            l2 = (x_adv - x).flatten(start_dim=1).norm(dim=1)
            f_val = self._attack_objective(self.model(x_adv), y)
            loss = (l2 + self.c * f_val).mean()
            loss.backward()
            optimizer.step()

            # Track best (lowest distortion) successful perturbation
            with torch.no_grad():
                successful = f_val <= 0
                improved = successful & (l2 < best_l2)
                x_best[improved] = x_adv[improved]
                best_l2[improved] = l2[improved]

        # Fall back to clean input if distortion exceeds budget
        within_budget = best_l2 <= self.epsilon
        result = x.clone()
        result[within_budget] = x_best[within_budget]

        self.model.train(training)
        return result.detach()
