"""
Evaluation metrics for adversarial robustness.

Definitions
-----------
Clean accuracy
    :math:`\\operatorname{Acc}(f, \\mathcal{D}) =
    \\frac{1}{N}\\sum_{i=1}^{N}
    \\mathbf{1}[\\hat{y}_i = y_i]`

Adversarial accuracy
    :math:`\\operatorname{Acc}_{\\mathrm{adv}}(f, \\mathcal{A}, \\mathcal{D}) =
    \\frac{1}{N}\\sum_{i=1}^{N}
    \\mathbf{1}[f(x_i + \\delta_i^*) = y_i]`

Mean perturbation norm
    :math:`\\bar{\\|\\delta\\|}_p =
    \\frac{1}{N}\\sum_{i=1}^{N}\\|x_i^{\\mathrm{adv}} - x_i\\|_p`
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def clean_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the classification accuracy on clean (unperturbed) inputs.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    x : torch.Tensor
        Input batch of shape ``(N, *)``.
    y : torch.Tensor
        Ground-truth integer labels of shape ``(N,)``.

    Returns
    -------
    float
        Fraction of correctly classified examples in ``[0, 1]``.
    """
    model.eval()
    preds = model(x).argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def adversarial_accuracy(
    model: nn.Module,
    x_adv: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute classification accuracy on adversarial examples.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    x_adv : torch.Tensor
        Adversarial examples of shape ``(N, *)``.
    y : torch.Tensor
        Ground-truth integer labels of shape ``(N,)``.

    Returns
    -------
    float
        Fraction of adversarial examples that are still correctly
        classified (lower is better for the attacker).
    """
    return clean_accuracy(model, x_adv, y)


@torch.no_grad()
def mean_perturbation_norm(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    p: float = 2.0,
) -> float:
    """Compute the mean :math:`\\ell_p` norm of the adversarial perturbations.

    Parameters
    ----------
    x_clean : torch.Tensor
        Original clean images of shape ``(N, *)``.
    x_adv : torch.Tensor
        Adversarial images of the same shape.
    p : float
        Order of the norm.  Common choices: ``2.0`` (:math:`\\ell_2`),
        ``float("inf")`` (:math:`\\ell_\\infty`).

    Returns
    -------
    float
        Mean per-sample perturbation norm.
    """
    delta = (x_adv - x_clean).flatten(start_dim=1)
    if p == float("inf"):
        norms = delta.abs().max(dim=1).values
    else:
        norms = delta.norm(p=p, dim=1)
    return norms.mean().item()
