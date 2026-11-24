"""Plane-stress finite element model for a compliant mechanism at large displacement.

Mesh numbering, the linear element matrix and the density filter are those of
Project 1 (top88.py). This module adds what Project 1 did not need: finite
strains, Newton's method, stresses, and adjoint sensitivities.

State equation (u nodal displacements, rho element densities):

    R(u; rho) = sum_e E_e(rho_e) f_e(u_e) + K_s u - F = 0.

Kinematics: total Lagrangian, Green-Lagrange strain E = (H + H^T + H^T H)/2 with
H = grad u, St. Venant-Kirchhoff material S = C E, internal force
f_e = integral of B(u)^T S dV. SIMP modulus E_e = E_min + rho^p (E_0 - E_min),
as in Project 1. K_s holds the spring of the gripped object.

Elements with density below rho_switch carry less than 1% of the solid stiffness.
They use the linearized strain, f_e = k_0 u_e, because near-void elements distort
so much at large displacement that Newton's method fails on them (Wang, Lazarov
and Sigmund 2014 use a smooth version of this switch). With nonlinear=False every
element is linear and the model is that of Project 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "01_problem_formulation" / "src"))
from top88 import element_dofs, element_stiffness, filter_matrix  # noqa: E402,F401


class Mechanism:
    """nelx x nely square elements of side a (m), thickness t (m), top88 numbering."""

    def __init__(self, nelx, nely, a, t, E0, nu, p=3.0, E_min=1e-6, nonlinear=True, rho_switch=0.2):
        self.nelx, self.nely, self.a, self.t, self.rho_switch = nelx, nely, a, t, rho_switch
        self.E0, self.nu, self.p, self.E_min, self.nonlinear = E0, nu, p, E_min * E0, nonlinear
        self.ne, self.ndof = nelx * nely, 2 * (nelx + 1) * (nely + 1)
        self.edof = element_dofs(nelx, nely)
        self.rows = np.repeat(self.edof, 8, axis=1).ravel()
        self.cols = np.tile(self.edof, (1, 8)).ravel()
        self.C = 1 / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])  # unit modulus
        gp = 1 / np.sqrt(3)
        # shape-function gradients (npts, 2, 4); node order of top88: LL, LR, UR, UL
        self.dN = np.array([self._grad(x, y) for x in (-gp, gp) for y in (-gp, gp)])
        self.dN0 = self._grad(0.0, 0.0)[None]                 # element centre
        self.w = np.full(4, a * a / 4) * t                    # volume weights
        self.KL = self._linear_stiffness()                    # unit-modulus linear element matrix
        ex, ey = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1), indexing="ij")
        self.xy = a * np.column_stack([ex.ravel(), (nely - ey).ravel()])
        self.springs = np.zeros(self.ndof)
        self.fixed = np.array([], dtype=int)

    def _grad(self, xi, eta):
        dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return np.array([dxi, deta]) * 2 / self.a

    def _linear_B(self, dN):
        B = np.zeros(dN.shape[:-2] + (3, 8))
        B[..., 0, 0::2] = dN[..., 0, :]
        B[..., 1, 1::2] = dN[..., 1, :]
        B[..., 2, 0::2] = dN[..., 1, :]
        B[..., 2, 1::2] = dN[..., 0, :]
        return B

    def _linear_stiffness(self):
        B = self._linear_B(self.dN)
        return np.einsum("g,gia,ij,gjb->ab", self.w, B, self.C, B)

    @property
    def free(self):
        return np.setdiff1d(np.arange(self.ndof), self.fixed)

    # ----- interpolations ---------------------------------------------------------------
    def modulus(self, rho):
        E = self.E_min + rho**self.p * (self.E0 - self.E_min)
        return E, self.p * rho ** (self.p - 1) * (self.E0 - self.E_min)

    # ----- element quantities --------------------------------------------------------------
    def _nonlinear(self, ue, dN, w, tangent=True):
        """Unit-modulus internal force, tangent and energy from Green-Lagrange strains."""
        H = np.einsum("eka,gja->egkj", ue.reshape(-1, 4, 2).transpose(0, 2, 1), dN)   # H[k, j] = du_k/dX_j
        F = H + np.eye(2)
        Egl = 0.5 * (H + H.transpose(0, 1, 3, 2) + np.einsum("egki,egkj->egij", H, H))
        Ev = np.stack([Egl[..., 0, 0], Egl[..., 1, 1], 2 * Egl[..., 0, 1]], axis=-1)
        S = Ev @ self.C
        # B_nl maps nodal displacements to the variation of Ev
        B = np.zeros(F.shape[:2] + (3, 8))
        for k in range(2):
            B[..., 0, k::2] = F[..., k, 0, None] * dN[None, :, 0, :]
            B[..., 1, k::2] = F[..., k, 1, None] * dN[None, :, 1, :]
            B[..., 2, k::2] = F[..., k, 0, None] * dN[None, :, 1, :] + F[..., k, 1, None] * dN[None, :, 0, :]
        f = np.einsum("g,egia,egi->ea", w, B, S)
        W = 0.5 * np.einsum("g,egi,egi->e", w, Ev, S)
        K = None
        if tangent:
            K = np.einsum("g,egia,ij,egjb->eab", w, B, self.C, B)
            St = np.stack([np.stack([S[..., 0], S[..., 2]], -1), np.stack([S[..., 2], S[..., 1]], -1)], -2)
            geo = np.einsum("g,gia,egij,gjb->eab", w, dN, St, dN)          # (grad N_a) . S . (grad N_b)
            K[:, 0::2, 0::2] += geo
            K[:, 1::2, 1::2] += geo
        return f, K, W, (Ev, S, B)

    def finite(self, rho):
        """Elements analysed with the finite-strain measure."""
        return (rho >= self.rho_switch) if self.nonlinear else np.zeros(self.ne, dtype=bool)

    def element(self, u, rho, tangent=True):
        """Unit-modulus element force, tangent and energy."""
        ue = u[self.edof]
        f = ue @ self.KL
        K = np.broadcast_to(self.KL, (self.ne, 8, 8)).copy() if tangent else None
        W = 0.5 * np.einsum("ea,ea->e", ue, f)
        nl = self.finite(rho)
        if nl.any():
            fn, Kn, Wn, _ = self._nonlinear(ue[nl], self.dN, self.w, tangent)
            f[nl], W[nl] = fn, Wn
            if tangent:
                K[nl] = Kn
        return f, K, W

    # ----- global quantities ---------------------------------------------------------------
    def assemble(self, ke):
        return coo_matrix((ke.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsc()

    def internal(self, u, rho):
        f = self.element(u, rho, tangent=False)[0]
        R = self.springs * u
        np.add.at(R, self.edof, self.modulus(rho)[0][:, None] * f)
        return R

    def tangent(self, u, rho):
        K = self.assemble(self.modulus(rho)[0][:, None, None] * self.element(u, rho)[1])
        return K + coo_matrix((self.springs, (np.arange(self.ndof), np.arange(self.ndof))),
                              shape=(self.ndof, self.ndof)).tocsc()

    def energy(self, u, rho, F):
        """Total potential energy; its gradient is the residual."""
        return float(self.modulus(rho)[0] @ self.element(u, rho, tangent=False)[2]
                     + 0.5 * self.springs @ u**2 - F @ u)

    def solve(self, rho, F, u0=None, tol=1e-9, maxit=80, steps=1):
        """Newton's method; the load is applied in `steps` increments.

        The step length is halved until the potential energy decreases (Armijo rule).
        The energy is the right measure of progress here: a full Newton step on a
        bending-dominated structure can raise the residual while it lowers the energy."""
        free = self.free
        u = np.zeros(self.ndof) if u0 is None else u0.copy()
        scale = np.linalg.norm(F[free])
        history = []
        for k in range(1, steps + 1):
            Fk = F * k / steps
            R = (self.internal(u, rho) - Fk)[free]
            history = [np.linalg.norm(R) / scale]
            for _ in range(maxit):
                lu = splu(self.tangent(u, rho)[free, :][:, free].tocsc())
                if history[-1] < tol:
                    break
                du = np.zeros(self.ndof)
                du[free] = -lu.solve(R)
                slope, Pi, step = R @ du[free], self.energy(u, rho, Fk), 1.0
                # the last term is the round-off of the energy: near the solution the decrease is below it
                while step > 1e-3 and not (self.energy(u + step * du, rho, Fk)
                                           <= Pi + 1e-4 * step * slope + 1e-12 * abs(Pi)):
                    step *= 0.5
                u = u + step * du
                R = (self.internal(u, rho) - Fk)[free]
                history.append(np.linalg.norm(R) / scale)
            else:
                raise RuntimeError(f"Newton did not converge: {history[-5:]}")
        return u, {"lu": lu, "history": history}

    # ----- responses and adjoints ------------------------------------------------------------
    def center_stress(self, u, rho):
        """Stress of the solid material at element centres (ne x 3), from the strain of the
        actual displacement, and its derivative w.r.t. u_e (ne x 3 x 8). Second Piola-Kirchhoff
        stress in finite-strain elements."""
        ue = u[self.edof]
        B = self._linear_B(self.dN0)[0]
        sig = self.E0 * (ue @ B.T) @ self.C
        dsig = np.broadcast_to(self.E0 * self.C @ B, (self.ne, 3, 8)).copy()
        nl = self.finite(rho)
        if nl.any():
            _, _, _, (Ev, S, Bn) = self._nonlinear(ue[nl], self.dN0, np.ones(1), tangent=False)
            sig[nl] = self.E0 * S[:, 0]
            dsig[nl] = self.E0 * np.einsum("ij,ejb->eib", self.C, Bn[:, 0])
        return sig, dsig

    def adjoint_gradient(self, u, rho, lu, dg_du):
        """dg/drho_e for a response g(u): solve K_T lam = -dg/du, then dg/drho_e = lam_e . dR/drho_e,
        with dR/drho_e = E'(rho_e) f_e(u_e). K_T is symmetric, so its factorization is reused."""
        lam = np.zeros(self.ndof)
        lam[self.free] = lu.solve(-dg_du[self.free])
        return self.modulus(rho)[1] * np.einsum("ea,ea->e", lam[self.edof], self.element(u, rho, tangent=False)[0])


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
