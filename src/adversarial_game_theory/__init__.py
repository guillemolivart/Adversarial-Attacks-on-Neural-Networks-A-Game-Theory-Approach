"""
Adversarial Attacks on Neural Networks: A Game Theory Approach
==============================================================

A research-oriented library implementing adversarial machine learning
through the lens of game theory. Provides formal, modular implementations
of neural network architectures, adversarial attacks, and game-theoretic
frameworks suitable for academic research.

Modules
-------
models
    Neural network architectures (MLP, CNN).
attacks
    Adversarial attack algorithms (FGSM, PGD, Carlini-Wagner).
games
    Game-theoretic frameworks (GAN, Nash equilibrium).
utils
    Evaluation metrics and visualisation helpers.
"""

from adversarial_game_theory import attacks, games, models, utils

__all__ = ["models", "attacks", "games", "utils"]
__version__ = "0.1.0"
