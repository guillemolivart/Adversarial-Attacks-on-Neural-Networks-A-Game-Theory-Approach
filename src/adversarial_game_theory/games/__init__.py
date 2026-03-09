"""
Game-theoretic frameworks.

Submodules
----------
base
    Abstract base class for all games.
gan
    Generative Adversarial Network (minimax two-player zero-sum game).
nash
    Nash equilibrium solver for finite matrix games.
"""

from adversarial_game_theory.games.base import TwoPlayerGame
from adversarial_game_theory.games.gan import GAN
from adversarial_game_theory.games.nash import NashEquilibriumSolver

__all__ = ["TwoPlayerGame", "GAN", "NashEquilibriumSolver"]
