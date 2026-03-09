"""
Visualization helpers for adversarial examples and training histories.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import torch


def plot_adversarial_examples(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    labels: Sequence[str] | None = None,
    num_samples: int = 5,
    title: str = "Adversarial Examples",
) -> plt.Figure:
    """Plot clean images alongside their adversarial counterparts.

    Produces a grid with two rows: the top row shows clean images and the
    bottom row shows the corresponding adversarial images.

    Parameters
    ----------
    x_clean : torch.Tensor
        Clean image batch of shape ``(N, C, H, W)`` or ``(N, H, W)``,
        values in ``[0, 1]``.
    x_adv : torch.Tensor
        Adversarial image batch of the same shape.
    labels : Sequence[str], optional
        Per-sample text annotations displayed as column titles.
    num_samples : int
        Number of samples to display (leftmost ``num_samples`` of the
        batch).  Defaults to ``5``.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    n = min(num_samples, x_clean.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    fig.suptitle(title)

    for i in range(n):
        for row, (img, row_label) in enumerate(
            [(x_clean[i], "Clean"), (x_adv[i], "Adversarial")]
        ):
            img_np = img.detach().cpu()
            if img_np.dim() == 3:
                img_np = img_np.permute(1, 2, 0).squeeze(-1)
            ax = axes[row, i] if n > 1 else axes[row]
            ax.imshow(img_np.numpy(), cmap="gray" if img_np.dim() == 2 else None, vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(row_label)
            if row == 0 and labels is not None:
                ax.set_title(labels[i], fontsize=8)

    plt.tight_layout()
    return fig


def plot_training_history(
    attacker_losses: Sequence[float],
    defender_losses: Sequence[float],
    title: str = "GAN Training History",
) -> plt.Figure:
    """Plot attacker and defender losses over training rounds.

    Parameters
    ----------
    attacker_losses : Sequence[float]
        Per-round attacker (generator) losses.
    defender_losses : Sequence[float]
        Per-round defender (discriminator) losses.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    rounds = range(1, len(attacker_losses) + 1)
    ax.plot(rounds, attacker_losses, label="Attacker (Generator)", color="tab:blue")
    ax.plot(rounds, defender_losses, label="Defender (Discriminator)", color="tab:orange")
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
