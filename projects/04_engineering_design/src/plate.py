"""Finite element model of a thin heat-spreader plate, with adjoint sensitivities.

State equation (temperature T in kelvin, thickness t per element):

    -div(k t grad T) + h (T - T_inf) + eps*sigma (T^4 - T_inf^4) = q    in the plate,
    zero heat flux on the plate edges.

Four-node bilinear elements on a regular grid. The radiation term makes the
discrete residual R(T; t) = 0 nonlinear; it is solved by Newton's method and
differentiated implicitly at the converged state.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m^2 K^4)


def gauss_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss rule on the reference square [-1, 1]^2."""
    x, w = np.polynomial.legendre.leggauss(n)
    xi, eta = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([xi.ravel(), eta.ravel()]), np.outer(w, w).ravel()


def shape(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Bilinear shape functions, node order (-,-), (+,-), (+,+), (-,+)."""
    return 0.25 * np.column_stack(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
    )


def shape_grad(xi: np.ndarray, eta: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Physical gradients of the shape functions, shape (npts, 2, 4)."""
    dxi = 0.25 * np.column_stack([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    deta = 0.25 * np.column_stack([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return np.stack([dxi * 2 / dx, deta * 2 / dy], axis=1)


def element_matrices(dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit conduction matrix (k t = 1) and unit surface matrix (h = 1) of one element."""
    pts, w = gauss_rule(2)
    N = shape(pts[:, 0], pts[:, 1])
    B = shape_grad(pts[:, 0], pts[:, 1], dx, dy)
    jac = dx * dy / 4
    k0 = np.einsum("g,gdi,gdj->ij", w * jac, B, B)
    m0 = np.einsum("g,gi,gj->ij", w * jac, N, N)
    return k0, m0


class Plate:
    """Rectangular plate [0, Lx] x [0, Ly] meshed with nx x ny elements."""

    def __init__(self, Lx, Ly, nx, ny, k, h, eps, T_inf):
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        self.k, self.h, self.eps, self.T_inf = k, h, eps, T_inf
        self.dx, self.dy = Lx / nx, Ly / ny
        self.area = self.dx * self.dy
        self.nn, self.ne = (nx + 1) * (ny + 1), nx * ny

        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        n00 = (i + j * (nx + 1)).ravel()
        self.edof = np.column_stack([n00, n00 + 1, n00 + nx + 2, n00 + nx + 1])
        xs, ys = np.meshgrid(np.linspace(0, Lx, nx + 1), np.linspace(0, Ly, ny + 1), indexing="xy")
        self.xy = np.column_stack([xs.ravel(), ys.ravel()])
        self.centers = self.xy[self.edof].mean(axis=1)

        self.k0, self.m0 = element_matrices(self.dx, self.dy)
        self.rows = np.repeat(self.edof, 4, axis=1).ravel()
        self.cols = np.tile(self.edof, (1, 4)).ravel()
        self.M = self._assemble(np.broadcast_to(self.m0, (self.ne, 4, 4)))

        # 3 x 3 Gauss points integrate N_i T_h^4 and N_i N_j T_h^3 exactly.
        pts, w = gauss_rule(3)
        self.Ng = shape(pts[:, 0], pts[:, 1])
        self.wg = w * self.area / 4
        self.gauss_ref = pts

    def _assemble(self, ke: np.ndarray) -> csc_matrix:
        return coo_matrix((ke.ravel(), (self.rows, self.cols)), shape=(self.nn, self.nn)).tocsc()

    # ----- loads -------------------------------------------------------------
    def load_uniform(self, q_el: np.ndarray) -> np.ndarray:
        """Nodal load from a heat flux q_el (W/m^2) that is constant in each element."""
        Q = np.zeros(self.nn)
        np.add.at(Q, self.edof, (q_el * self.area / 4)[:, None])
        return Q

    def load_function(self, q) -> np.ndarray:
        """Nodal load from a heat flux q(x, y), integrated with the 3 x 3 rule."""
        xg = self.centers[:, None, 0] + 0.5 * self.dx * self.gauss_ref[None, :, 0]
        yg = self.centers[:, None, 1] + 0.5 * self.dy * self.gauss_ref[None, :, 1]
        Q = np.zeros(self.nn)
        np.add.at(Q, self.edof, (q(xg, yg) * self.wg) @ self.Ng)
        return Q

    # ----- residual and tangent ------------------------------------------------
    def conduction(self, t_el: np.ndarray) -> csc_matrix:
        return self._assemble((self.k * t_el)[:, None, None] * self.k0)

    def radiation(self, T: np.ndarray) -> tuple[np.ndarray, csc_matrix]:
        """Radiation residual vector and its tangent matrix."""
        Tg = T[self.edof] @ self.Ng.T
        c = self.eps * SIGMA
        r = np.zeros(self.nn)
        np.add.at(r, self.edof, c * ((Tg**4 - self.T_inf**4) * self.wg) @ self.Ng)
        kr = c * np.einsum("eg,gi,gj->eij", 4 * Tg**3 * self.wg, self.Ng, self.Ng)
        return r, self._assemble(kr)

    def residual(self, T, t_el, Q, Kc=None):
        """R(T; t). The linear terms act on theta = T - T_inf, which is exact
        (a uniform temperature conducts no heat) and keeps round-off small."""
        Kc = self.conduction(t_el) if Kc is None else Kc
        theta = T - self.T_inf
        R = Kc @ theta + self.h * (self.M @ theta) - Q
        if self.eps > 0:
            R = R + self.radiation(T)[0]
        return R

    def tangent(self, T, t_el, Kc=None):
        """K_T = dR/dT, symmetric positive definite."""
        Kc = self.conduction(t_el) if Kc is None else Kc
        KT = Kc + self.h * self.M
        return (KT + self.radiation(T)[1] if self.eps > 0 else KT).tocsc()

    def potential(self, T, t_el, Q):
        """Convex potential whose gradient is the residual R(T; t)."""
        theta = T - self.T_inf
        Tg = T[self.edof] @ self.Ng.T
        radiation = self.eps * SIGMA * np.sum((Tg**5 / 5 - self.T_inf**4 * Tg) * self.wg)
        return (0.5 * theta @ (self.conduction(t_el) @ theta) + 0.5 * self.h * theta @ (self.M @ theta)
                + radiation - Q @ theta)

    def solve(self, t_el, Q, T0=None, tol=1e-10, maxit=30):
        """Newton's method on R(T; t) = 0. Returns T and a dict with the
        factorized tangent at the solution and the residual history."""
        Kc = self.conduction(t_el)
        T = np.full(self.nn, self.T_inf) if T0 is None else T0.copy()
        scale = max(np.linalg.norm(Q), 1e-30)
        history = []
        for _ in range(maxit):
            R = self.residual(T, t_el, Q, Kc)
            history.append(np.linalg.norm(R) / scale)
            lu = splu(self.tangent(T, t_el, Kc))
            if history[-1] < tol:
                return T, {"lu": lu, "history": history}
            T = T - lu.solve(R)
        raise RuntimeError(f"Newton did not converge: {history}")

    # ----- responses and adjoints ----------------------------------------------
    def heat_out(self, T: np.ndarray) -> float:
        """Total heat leaving the surface by convection and radiation (W)."""
        out = self.h * np.sum(self.M @ (T - self.T_inf))
        if self.eps > 0:
            out += np.sum(self.radiation(T)[0])
        return float(out)

    def adjoint_gradient(self, T, lu, dg_dT) -> np.ndarray:
        """dg/dt_e for a response g(T) with no explicit thickness dependence.

        Solve K_T^T lam = -dg/dT, then dg/dt_e = lam_e^T (k k0) T_e, because
        dR/dt_e = k k0 T_e. K_T is symmetric, so the factorization is reused."""
        lam = lu.solve(-dg_dT)
        return self.k * np.einsum("ei,ij,ej->e", lam[self.edof], self.k0, T[self.edof])


def ks(a: np.ndarray, rho: float) -> tuple[float, np.ndarray]:
    """Kreisselmeier-Steinhauser aggregate of a and its derivative (softmax weights)."""
    amax = a.max()
    e = np.exp(rho * (a - amax))
    return amax + np.log(e.sum()) / rho, e / e.sum()
