"""
Adversarial attack algorithms.

Submodules
----------
base
    Abstract base class for all attacks.
fgsm
    Fast Gradient Sign Method (Goodfellow et al., 2015).
pgd
    Projected Gradient Descent (Madry et al., 2018).
carlini_wagner
    Carlini & Wagner L2 attack (Carlini & Wagner, 2017).
"""

from adversarial_game_theory.attacks.base import AdversarialAttack
from adversarial_game_theory.attacks.carlini_wagner import CarliniWagnerL2
from adversarial_game_theory.attacks.fgsm import FGSM
from adversarial_game_theory.attacks.pgd import PGD

__all__ = ["AdversarialAttack", "FGSM", "PGD", "CarliniWagnerL2"]
