"""Method of moving asymptotes (Svanberg 1987), with the subproblem solved through its dual.

Minimize f0(x) subject to f_i(x) <= 0 and bounds. Each function is replaced by
a separable convex approximation p_j/(U_j - x_j) + q_j/(x_j - L_j). For given
multipliers the minimizer of the subproblem is known in closed form, so the
dual has only as many unknowns as there are constraints.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class MMA:
    def __init__(self, n, m, x_min=0.0, x_max=1.0, move=0.2, c=1000.0):
        self.n, self.m, self.x_min, self.x_max, self.move, self.c = n, m, x_min, x_max, move, c
        self.history = []          # previous two iterates, for the asymptote update
        self.low = self.upp = None

    def step(self, x, f0, df0, f, df):
        """One MMA update. f (m,), df (m, n). Returns the new x and the multipliers."""
        span = self.x_max - self.x_min
        if len(self.history) < 2:
            self.low, self.upp = x - 0.5 * span, x + 0.5 * span
        else:
            x1, x2 = self.history[-1], self.history[-2]
            sign = (x - x1) * (x1 - x2)
            factor = np.where(sign > 0, 1.2, np.where(sign < 0, 0.7, 1.0))
            self.low = np.clip(x - factor * (x1 - self.low), x - 10 * span, x - 0.01 * span)
            self.upp = np.clip(x + factor * (self.upp - x1), x + 0.01 * span, x + 10 * span)
        L, U = self.low, self.upp
        alpha = np.maximum.reduce([np.full(self.n, self.x_min), L + 0.1 * (x - L), x - self.move * span])
        beta = np.minimum.reduce([np.full(self.n, self.x_max), U - 0.1 * (U - x), x + self.move * span])

        def pq(g):
            pos, neg = np.maximum(g, 0.0), np.maximum(-g, 0.0)
            reg = 1e-5 / span
            return (U - x) ** 2 * (1.001 * pos + 0.001 * neg + reg), (x - L) ** 2 * (0.001 * pos + 1.001 * neg + reg)

        p0, q0 = pq(df0)
        P, Q = map(np.array, zip(*(pq(g) for g in df)))
        r = f - (P / (U - x) + Q / (x - L)).sum(axis=1)

        def primal(lam):
            p, q = p0 + lam @ P, q0 + lam @ Q
            return np.clip((np.sqrt(p) * L + np.sqrt(q) * U) / (np.sqrt(p) + np.sqrt(q)), alpha, beta)

        def dual(lam):                       # negative dual function and its gradient
            xs = primal(lam)
            g = (P / (U - xs) + Q / (xs - L)).sum(axis=1) + r
            value = (p0 / (U - xs) + q0 / (xs - L)).sum() + lam @ g
            return -value, -g

        res = minimize(dual, np.ones(self.m), jac=True, method="L-BFGS-B", bounds=[(0.0, self.c)] * self.m,
                       options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500})
        self.history = (self.history + [x.copy()])[-2:]
        return primal(res.x), res.x
