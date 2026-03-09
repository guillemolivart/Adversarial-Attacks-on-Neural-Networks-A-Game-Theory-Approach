"""
Generative Adversarial Network (GAN) as a minimax two-player zero-sum game.

Mathematical definition
-----------------------
A GAN (Goodfellow et al., 2014) is a two-player zero-sum game between a
**generator** :math:`G_\\phi : \\mathcal{Z} \\to \\mathcal{X}` and a
**discriminator** :math:`D_\\psi : \\mathcal{X} \\to [0, 1]`.  The
minimax objective is

.. math::

    \\min_\\phi\\, \\max_\\psi\\;
    V(G_\\phi, D_\\psi) =
        \\mathbb{E}_{x \\sim p_{\\mathrm{data}}}[\\log D_\\psi(x)]
        + \\mathbb{E}_{z \\sim p_z}[\\log(1 - D_\\psi(G_\\phi(z)))].

The Nash equilibrium is achieved when :math:`p_G = p_{\\mathrm{data}}`
and :math:`D_\\psi(x) = \\tfrac{1}{2}` everywhere, at which point the
optimal discriminator loss equals :math:`\\log 4` and the generator
loss equals :math:`-\\log 4`.

Training alternates between:

1. **Discriminator step** — maximise :math:`V` w.r.t. :math:`\\psi`:

   .. math::

       \\mathcal{L}_D = -\\mathbb{E}[\\log D(x)] - \\mathbb{E}[\\log(1-D(G(z)))]

2. **Generator step** — minimise :math:`V` w.r.t. :math:`\\phi` using
   the non-saturating heuristic:

   .. math::

       \\mathcal{L}_G = -\\mathbb{E}[\\log D(G(z))]

References
----------
Goodfellow, I. J., et al. (2014).
*Generative Adversarial Nets.*
NeurIPS 2014. https://arxiv.org/abs/1406.2661
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from adversarial_game_theory.games.base import GameHistory, TwoPlayerGame


class GAN(TwoPlayerGame):
    """Generative Adversarial Network implemented as a minimax game.

    Parameters
    ----------
    generator : nn.Module
        The generator network :math:`G_\\phi`.  Must accept noise tensors
        of shape ``(batch_size, latent_dim)`` and return images of shape
        ``(batch_size, *image_shape)``.
    discriminator : nn.Module
        The discriminator network :math:`D_\\psi`.  Must accept images of
        shape ``(batch_size, *image_shape)`` and return logits of shape
        ``(batch_size, 1)``.
    latent_dim : int
        Dimensionality of the noise prior :math:`z \\sim p_z`.
    generator_lr : float
        Learning rate for the generator Adam optimiser.
    discriminator_lr : float
        Learning rate for the discriminator Adam optimiser.
    device : torch.device, optional
        Target compute device.

    Examples
    --------
    >>> G = MLP(latent_dim, [256, 512], image_dim)
    >>> D = MLP(image_dim, [512, 256], 1)
    >>> gan = GAN(G, D, latent_dim=100)
    >>> history = gan.train(num_rounds=1000, dataloader=loader)
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        latent_dim: int,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(name="GAN")
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.device = device or torch.device("cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
        self._bce = nn.BCEWithLogitsLoss()

    def play_round(self, data: torch.Tensor) -> tuple[float, float]:
        """Execute one GAN training round.

        Performs one discriminator update followed by one generator update.

        Parameters
        ----------
        data : torch.Tensor
            Batch of real images of shape ``(N, *image_shape)``.

        Returns
        -------
        tuple[float, float]
            ``(generator_loss, discriminator_loss)`` for this round.
        """
        real = data.to(self.device)
        n = real.size(0)

        real_labels = torch.ones(n, 1, device=self.device)
        fake_labels = torch.zeros(n, 1, device=self.device)

        # ── Discriminator step ──────────────────────────────────────────
        self.d_optimizer.zero_grad()
        z = torch.randn(n, self.latent_dim, device=self.device)
        fake = self.generator(z).detach()

        d_loss_real = self._bce(self.discriminator(real), real_labels)
        d_loss_fake = self._bce(self.discriminator(fake), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # ── Generator step (non-saturating) ─────────────────────────────
        self.g_optimizer.zero_grad()
        z = torch.randn(n, self.latent_dim, device=self.device)
        fake = self.generator(z)
        g_loss = self._bce(self.discriminator(fake), real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def train(self, num_rounds: int, dataloader: Any, **kwargs: Any) -> GameHistory:
        """Train the GAN for ``num_rounds`` data batches.

        Parameters
        ----------
        num_rounds : int
            Number of training rounds (one batch per round).
        dataloader : Iterable
            PyTorch DataLoader (or any iterable) yielding batches of real
            images.  Iterated cyclically if ``num_rounds`` exceeds the
            dataset size.
        **kwargs
            Ignored; present for API compatibility.

        Returns
        -------
        GameHistory
            Training history containing per-round generator and
            discriminator losses.
        """
        self.history = GameHistory()
        data_iter = iter(dataloader)

        for _ in range(num_rounds):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # DataLoader typically returns (images, labels); take images
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            g_loss, d_loss = self.play_round(batch)
            self.history.attacker_losses.append(g_loss)
            self.history.defender_losses.append(d_loss)

        return self.history

    @torch.no_grad()
    def generate(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic samples from the trained generator.

        Parameters
        ----------
        num_samples : int
            Number of images to generate.

        Returns
        -------
        torch.Tensor
            Generated images of shape ``(num_samples, *image_shape)``.
        """
        self.generator.eval()
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        return self.generator(z)
