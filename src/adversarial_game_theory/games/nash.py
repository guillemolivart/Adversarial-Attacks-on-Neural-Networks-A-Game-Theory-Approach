"""
Nash equilibrium solver for finite two-player zero-sum matrix games.

Mathematical definition
-----------------------
A **finite two-player zero-sum game** is characterised by a payoff matrix
:math:`A \\in \\mathbb{R}^{m \\times n}`, where the row player (attacker)
chooses a row :math:`i \\in \\{1,\\dots,m\\}` and the column player
(defender) chooses a column :math:`j \\in \\{1,\\dots,n\\}`.  The
attacker receives payoff :math:`A_{ij}` and the defender receives
:math:`-A_{ij}`.

By von Neumann's **minimax theorem** (1928), there exists a unique
**value** :math:`v^*` and **Nash equilibrium** mixed strategies
:math:`(p^*, q^*)` such that

.. math::

    v^* = \\max_{p \\in \\Delta_m}\\min_{q \\in \\Delta_n}\\; p^\\top A q
         = \\min_{q \\in \\Delta_n}\\max_{p \\in \\Delta_m}\\; p^\\top A q,

where :math:`\\Delta_k = \\{u \\in \\mathbb{R}^k : u \\ge 0,\\
\\mathbf{1}^\\top u = 1\\}` is the probability simplex.

**Implementation** — the problem is reduced to a linear programme via the
standard transformation (Dantzig, 1951):

Attacker's LP
    .. math::

        \\max_{p,\\, v}\\; v \\quad \\text{s.t.} \\quad
        A^\\top p \\ge v \\mathbf{1},\\;
        p \\ge 0,\\; \\mathbf{1}^\\top p = 1.

Solved here via :func:`scipy.optimize.linprog`.

References
----------
von Neumann, J. (1928). Zur Theorie der Gesellschaftsspiele.
*Mathematische Annalen*, 100, 295–320.

Dantzig, G. B. (1951). *Maximization of a linear function of variables
subject to linear inequalities.* Activity Analysis of Production and
Allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog


@dataclass(frozen=True)
class NashEquilibrium:
    """Solution to a finite two-player zero-sum game.

    Attributes
    ----------
    attacker_strategy : np.ndarray
        Optimal mixed strategy :math:`p^* \\in \\Delta_m` for the attacker
        (row player).  Shape ``(m,)``.
    defender_strategy : np.ndarray
        Optimal mixed strategy :math:`q^* \\in \\Delta_n` for the defender
        (column player).  Shape ``(n,)``.
    value : float
        Game value :math:`v^* = (p^*)^\\top A\\, q^*`.
    """

    attacker_strategy: npt.NDArray[np.float64]
    defender_strategy: npt.NDArray[np.float64]
    value: float


class NashEquilibriumSolver:
    """Solve finite two-player zero-sum games via linear programming.

    Parameters
    ----------
    payoff_matrix : array-like of shape (m, n)
        Payoff matrix :math:`A` where entry :math:`A_{ij}` is the
        attacker's payoff when the attacker plays row :math:`i` and the
        defender plays column :math:`j`.

    Examples
    --------
    >>> A = np.array([[3, -1], [-2, 4]], dtype=float)
    >>> solver = NashEquilibriumSolver(A)
    >>> eq = solver.solve()
    >>> eq.value          # game value
    >>> eq.attacker_strategy  # optimal mixed strategy for the attacker
    """

    def __init__(self, payoff_matrix: npt.ArrayLike) -> None:
        self._A: npt.NDArray[np.float64] = np.asarray(payoff_matrix, dtype=np.float64)
        if self._A.ndim != 2:
            raise ValueError(f"payoff_matrix must be 2-D, got shape {self._A.shape}.")

    @property
    def payoff_matrix(self) -> npt.NDArray[np.float64]:
        """The payoff matrix :math:`A`."""
        return self._A

    def solve(self) -> NashEquilibrium:
        """Compute a Nash equilibrium via linear programming.

        Returns
        -------
        NashEquilibrium
            Optimal mixed strategies for both players and the game value.

        Raises
        ------
        RuntimeError
            If the linear programme fails to find a solution.
        """
        m, n = self._A.shape

        # Shift to make all entries positive (LP requires v > 0)
        shift = -self._A.min() + 1.0
        A_shifted = self._A + shift

        # ── Attacker's LP ────────────────────────────────────────────────
        # Variables: (p_1, …, p_m, v)
        # Maximise v  ↔  minimise -v
        # Subject to:
        #   A^T p ≥ v 1  →  -A^T p + v 1 ≤ 0
        #   1^T p = 1
        #   p ≥ 0
        c = np.zeros(m + 1)
        c[-1] = -1.0  # minimise -v

        # Inequality: -A^T p + v ≤ 0  (n constraints, m+1 variables)
        A_ub = np.hstack([-A_shifted.T, np.ones((n, 1))])
        b_ub = np.zeros(n)

        # Equality: 1^T p = 1
        A_eq = np.ones((1, m + 1))
        A_eq[0, -1] = 0.0
        b_eq = np.array([1.0])

        # Bounds: p_i ≥ 0, v unbounded (but positive after shift)
        bounds = [(0.0, None)] * m + [(None, None)]

        result_p = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if result_p.status != 0:
            raise RuntimeError(f"Attacker LP failed: {result_p.message}")

        p_star = result_p.x[:m]
        p_star = np.clip(p_star, 0.0, None)
        p_star /= p_star.sum()
        v_shifted = float(result_p.x[-1])

        # ── Defender's LP (dual) ─────────────────────────────────────────
        # Variables: (q_1, …, q_n, v)
        # Minimise v  →  from the shifted problem
        c_d = np.zeros(n + 1)
        c_d[-1] = 1.0

        A_ub_d = np.hstack([A_shifted, -np.ones((m, 1))])
        b_ub_d = np.zeros(m)

        A_eq_d = np.ones((1, n + 1))
        A_eq_d[0, -1] = 0.0
        b_eq_d = np.array([1.0])

        bounds_d = [(0.0, None)] * n + [(None, None)]

        result_q = linprog(c_d, A_ub=A_ub_d, b_ub=b_ub_d, A_eq=A_eq_d, b_eq=b_eq_d, bounds=bounds_d, method="highs")
        if result_q.status != 0:
            raise RuntimeError(f"Defender LP failed: {result_q.message}")

        q_star = result_q.x[:n]
        q_star = np.clip(q_star, 0.0, None)
        q_star /= q_star.sum()

        # Remove the shift from the game value
        value = v_shifted - shift

        return NashEquilibrium(
            attacker_strategy=p_star,
            defender_strategy=q_star,
            value=value,
        )
