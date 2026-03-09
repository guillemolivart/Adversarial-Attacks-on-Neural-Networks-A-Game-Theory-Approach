"""
Abstract base class for two-player games.

Formal definition
-----------------
A **two-player game** :math:`\\Gamma` is a tuple

.. math::

    \\Gamma = (\\mathcal{A},\\, \\mathcal{D},\\, u_A,\\, u_D),

where

* :math:`\\mathcal{A}` — strategy space of the **attacker** (player 1),
* :math:`\\mathcal{D}` — strategy space of the **defender** (player 2),
* :math:`u_A : \\mathcal{A} \\times \\mathcal{D} \\to \\mathbb{R}` —
  attacker's utility function,
* :math:`u_D : \\mathcal{A} \\times \\mathcal{D} \\to \\mathbb{R}` —
  defender's utility function.

The game is **zero-sum** when :math:`u_A + u_D = 0` for all strategy
pairs, and the Nash equilibrium satisfies

.. math::

    u_A(a^*, d^*) \\ge u_A(a, d^*) \\quad \\forall a \\in \\mathcal{A}, \\\\
    u_D(a^*, d^*) \\ge u_D(a^*, d) \\quad \\forall d \\in \\mathcal{D}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameHistory:
    """Container for training metrics collected during a game.

    Attributes
    ----------
    attacker_losses : list[float]
        Per-iteration loss values for the attacker / generator.
    defender_losses : list[float]
        Per-iteration loss values for the defender / discriminator.
    """

    attacker_losses: list[float] = field(default_factory=list)
    defender_losses: list[float] = field(default_factory=list)


class TwoPlayerGame(ABC):
    """Abstract base class for two-player adversarial games.

    Concrete subclasses must implement :meth:`play_round` (a single
    interaction step) and :meth:`train` (the full iterative procedure).

    Parameters
    ----------
    name : str
        Human-readable name of the game (e.g. ``"GAN"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.history: GameHistory = GameHistory()

    @abstractmethod
    def play_round(self, data: Any) -> tuple[float, float]:
        """Execute one round of the game.

        Parameters
        ----------
        data : Any
            A batch of real data used in this round.

        Returns
        -------
        tuple[float, float]
            ``(attacker_loss, defender_loss)`` for the current round.
        """

    @abstractmethod
    def train(self, num_rounds: int, **kwargs: Any) -> GameHistory:
        """Run the full training loop for ``num_rounds`` rounds.

        Parameters
        ----------
        num_rounds : int
            Total number of game rounds (training iterations).
        **kwargs
            Additional keyword arguments passed to the concrete
            implementation.

        Returns
        -------
        GameHistory
            Collected loss history.
        """
