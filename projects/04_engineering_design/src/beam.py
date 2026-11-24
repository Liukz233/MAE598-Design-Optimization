"""Plane-stress finite element model of the half MBB beam with a nonlinear material.

The mesh, the element numbering and the supports are those of Project 1
(top88.py); this module adds what Project 1 did not need: stresses, a nonlinear
stress-strain law, Newton's method, and adjoint sensitivities.

State equation (u nodal displacements, t thickness per element):

    R(u; t) = sum_e t_e * f_e(u_e) - F = 0,   f_e = integral of B^T sigma(eps) dA.

Material: sigma = g C eps, where C is the plane-stress matrix and the secant
factor g follows the Ramberg-Osgood curve eps = s/E + 0.002 (s/f_y)^n in terms
of the energy-equivalent stress s_e = sqrt(E eps^T C eps). With alpha = 0 the
law is linear (g = 1) and the model is the one of Project 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01_problem_formulation" / "src"))
from top88 import element_dofs, element_stiffness, half_mbb_load_and_supports  # noqa: E402,F401


def shape_gradients(xi: float, eta: float, a: float) -> np.ndarray:
    """Strain-displacement matrix B (3 x 8) of a square element of side a.

    Node order as in top88: lower left, lower right, upper right, upper left."""
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) * 2 / a
    deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) * 2 / a
    B = np.zeros((3, 8))
    B[0, 0::2], B[1, 1::2] = dxi, deta
    B[2, 0::2], B[2, 1::2] = deta, dxi
    return B


class Beam:
    """Half MBB beam: nelx x nely square elements of side a (m)."""

    def __init__(self, nelx, nely, a, E, nu, f_y=1.0, n=6.0, nonlinear=True):
        self.nelx, self.nely, self.a = nelx, nely, a
        self.E, self.nu, self.f_y, self.n = E, nu, f_y, n
        self.alpha = 0.002 * E / f_y if nonlinear else 0.0
        self.ne, self.ndof = nelx * nely, 2 * (nelx + 1) * (nely + 1)
        self.edof = element_dofs(nelx, nely)
        self.free = half_mbb_load_and_supports(nelx, nely)[1]  # symmetry edge and roller, as in Project 1
        self.rows = np.repeat(self.edof, 8, axis=1).ravel()
        self.cols = np.tile(self.edof, (1, 8)).ravel()
        self.C = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        gp = 1 / np.sqrt(3)
        self.Bg = np.array([shape_gradients(x, y, a) for x in (-gp, gp) for y in (-gp, gp)])
        self.wg = np.full(4, a * a / 4)
        self.Bc = shape_gradients(0.0, 0.0, a)  # element centre, where stresses are reported
        # node coordinates: top88 numbers nodes column by column, starting at the top
        ex, ey = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1), indexing="ij")
        self.xy = a * np.column_stack([ex.ravel(), (nely - ey).ravel()])

    # ----- material -----------------------------------------------------------------
    def material(self, eps):
        """Stress, tangent matrix and energy density for strains eps (..., 3)."""
        s = eps @ self.C                                   # C eps
        q = np.einsum("...i,...i->...", eps, s)            # eps^T C eps
        if self.alpha == 0.0:
            return s, np.broadcast_to(self.C, eps.shape[:-1] + (3, 3)), 0.5 * q
        ze = np.sqrt(self.E * np.maximum(q, 0.0)) / self.f_y
        z = np.minimum(ze, (ze / self.alpha) ** (1 / self.n))   # starts above the root
        for _ in range(60):                                # Newton on z + alpha z^n = ze
            dz = (z + self.alpha * z**self.n - ze) / (1 + self.alpha * self.n * z ** (self.n - 1))
            z = z - dz
            if np.max(np.abs(dz)) < 1e-15:
                break
        small = ze < 1e-12
        g = np.where(small, 1.0, z / np.where(small, 1.0, ze))
        slope = 1 / (1 + self.alpha * self.n * z ** (self.n - 1))
        D = g[..., None, None] * self.C + ((slope - g) / np.where(small, 1.0, q))[..., None, None] * (
            s[..., :, None] * s[..., None, :])
        W = self.f_y**2 / self.E * (z * ze - 0.5 * z**2 - self.alpha * z ** (self.n + 1) / (self.n + 1))
        return g[..., None] * s, D, W

    # ----- residual, tangent, energy ---------------------------------------------------
    def element_forces(self, u, tangent=False):
        """Internal force per unit thickness of every element, and its tangent."""
        eps = np.einsum("gij,ej->egi", self.Bg, u[self.edof])
        sig, D, W = self.material(eps)
        f = np.einsum("g,gij,egi->ej", self.wg, self.Bg, sig)
        k = np.einsum("g,gia,egij,gjb->eab", self.wg, self.Bg, D, self.Bg) if tangent else None
        return f, k, W @ self.wg

    def assemble(self, ke):
        return coo_matrix((ke.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsc()

    def internal(self, u, t_el):
        f, _, _ = self.element_forces(u)
        R = np.zeros(self.ndof)
        np.add.at(R, self.edof, t_el[:, None] * f)
        return R

    def tangent(self, u, t_el):
        _, k, _ = self.element_forces(u, tangent=True)
        return self.assemble(t_el[:, None, None] * k)

    def energy(self, u, t_el, F):
        """Total potential energy; its gradient is the residual."""
        return float(t_el @ self.element_forces(u)[2] - F @ u)

    def solve(self, t_el, F, u0=None, tol=1e-9, maxit=40):
        """Newton's method with step halving on the free degrees of freedom."""
        free = self.free
        u = np.zeros(self.ndof) if u0 is None else u0.copy()
        scale = np.linalg.norm(F[free])
        R = (self.internal(u, t_el) - F)[free]
        history = [np.linalg.norm(R) / scale]
        for _ in range(maxit):
            lu = splu(self.tangent(u, t_el)[free, :][:, free].tocsc())
            if history[-1] < tol:
                return u, {"lu": lu, "history": history}
            du = np.zeros(self.ndof)
            du[free] = -lu.solve(R)
            step = 1.0
            while True:
                trial = u + step * du
                R_trial = (self.internal(trial, t_el) - F)[free]
                if np.linalg.norm(R_trial) / scale < history[-1] or step < 1e-3:
                    break
                step *= 0.5
            u, R = trial, R_trial
            history.append(np.linalg.norm(R) / scale)
        raise RuntimeError(f"Newton did not converge: {history[-5:]}")

    # ----- responses and adjoints ----------------------------------------------------
    def center_stress(self, u):
        """Stress at element centres (ne x 3) and its derivative w.r.t. u_e (ne x 3 x 8)."""
        sig, D, _ = self.material(u[self.edof] @ self.Bc.T)
        return sig, D @ self.Bc

    def adjoint_gradient(self, u, lu, dg_du):
        """dg/dt_e for a response g(u): solve K_T lam = -dg/du, then dg/dt_e = lam_e . f_e,
        because dR/dt_e = f_e(u_e). K_T is symmetric, so its factorization is reused."""
        lam = np.zeros(self.ndof)
        lam[self.free] = lu.solve(-dg_du[self.free])
        return np.einsum("ei,ei->e", lam[self.edof], self.element_forces(u)[0])


def von_mises(sig):
    """Plane-stress von Mises stress and its derivative w.r.t. (sx, sy, txy)."""
    sx, sy, txy = sig.T
    vm = np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)
    return vm, np.column_stack([2 * sx - sy, 2 * sy - sx, 6 * txy]) / (2 * np.maximum(vm, 1e-30))[:, None]


def ks(a, rho):
    """Kreisselmeier-Steinhauser aggregate of a and its derivative (softmax weights)."""
    amax = a.max()
    e = np.exp(rho * (a - amax))
    return amax + np.log(e.sum()) / rho, e / e.sum()
