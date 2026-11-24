"""Design problem: a compliant gripper as a material layout on the Project 1 mesh.

Half of the gripper is modelled (it is symmetric about its centre line, the
bottom edge of the mesh). Design variable x_e in [0, 1] per element, filtered
with the Project 1 density filter to give the physical density rho_e. SI units.
"""

from __future__ import annotations

import numpy as np

from mech import Mechanism, filter_matrix, ks, von_mises

CONFIG = {
    # geometry of the half gripper
    "length": 0.100, "half_height": 0.050, "thickness": 0.005,   # m
    "nelx": 60, "nely": 30,
    "gap_length": 0.020, "gap_half_height": 0.010,              # jaw opening, kept empty
    "support_height": 0.010,                                    # clamped part of the left edge, from the top
    "pad": 0.005,                                               # m, solid pads at the ports and the clamp
    # polypropylene
    "E": 1.4e9, "nu": 0.4, "rho": 900.0, "S_y": 34e6,
    # actuator and workpiece
    "F_in": 20.0,                # N, whole gripper
    "stroke_max": 5.0e-3,        # m, actuator stroke
    "k_out": 2.0e3,              # N/m, stiffness of the gripped object, per jaw
    # SIMP and filter, as in Project 1
    "penal": 3.0, "E_min": 1e-6, "rmin": 2.4, "volume_fraction": 0.30,
    # stress constraint
    "safety_factor": 1.5, "stress_exponent": 1.0, "ks_rho": 40.0,
}


U_REF = 1.0e-3   # m; the objective is the jaw closing in millimetres


class Gripper:
    """Analysis, responses and adjoint gradients for one kinematic model."""

    names = ("volume", "stroke", "stress")

    def __init__(self, cfg=CONFIG, nonlinear=True, nelx=None, nely=None):
        self.cfg, self.nonlinear, c = cfg, nonlinear, cfg
        nelx, nely = nelx or c["nelx"], nely or c["nely"]
        a = c["length"] / nelx
        self.m = m = Mechanism(nelx, nely, a, c["thickness"], c["E"], c["nu"], c["penal"], c["E_min"], nonlinear)
        self.n = m.ne
        x, y = m.xy.T
        L, H = c["length"], c["half_height"]
        cx = a * (np.arange(m.ne) // nely + 0.5)                       # element centres, top88 numbering
        cy = a * (nely - np.arange(m.ne) % nely - 0.5)
        pad = c["pad"]
        # passive regions
        self.void = (cx > L - c["gap_length"]) & (cy < c["gap_half_height"])
        in_pad = (cx < pad) & (cy < pad)
        out_pad = (cx > L - pad) & (cy > c["gap_half_height"]) & (cy < c["gap_half_height"] + pad)
        support_pad = (cx < pad) & (cy > H - c["support_height"])
        self.solid = in_pad | out_pad | support_pad
        self.active = ~(self.void | self.solid)
        # supports: clamp at the top of the left edge, symmetry on the centre line outside the jaw opening
        tol = 1e-9
        clamp = np.flatnonzero((x < tol) & (y > H - c["support_height"] - tol))
        sym = np.flatnonzero((y < tol) & (x < L - c["gap_length"] + tol))
        m.fixed = np.unique(np.concatenate([2 * clamp, 2 * clamp + 1, 2 * sym + 1]))
        # input: half the actuator force pushes the pad on the centre line in +x
        self.in_nodes = np.flatnonzero((x < tol) & (y < pad + tol))
        self.F = np.zeros(m.ndof)
        w = np.ones(self.in_nodes.size)
        w[[np.argmin(y[self.in_nodes]), np.argmax(y[self.in_nodes])]] = 0.5
        self.F[2 * self.in_nodes] = 0.5 * c["F_in"] * w / w.sum()
        self.in_weights = np.zeros(m.ndof)
        self.in_weights[2 * self.in_nodes] = w / w.sum()               # stroke = weighted mean of u_x
        # output: closing motion (-y) of the jaw face, resisted by the workpiece spring
        self.out_nodes = np.flatnonzero((np.abs(y - c["gap_half_height"]) < tol) & (x > L - pad - tol))
        wo = np.ones(self.out_nodes.size)
        wo[[np.argmin(x[self.out_nodes]), np.argmax(x[self.out_nodes])]] = 0.5
        m.springs[2 * self.out_nodes + 1] = c["k_out"] * wo / wo.sum()
        self.out_weights = np.zeros(m.ndof)
        self.out_weights[2 * self.out_nodes + 1] = -wo / wo.sum()      # closing = mean of -u_y
        # density filter of Project 1
        self.Hf, self.Hs = filter_matrix(nelx, nely, c["rmin"])
        self.sigma_allow = c["S_y"] / c["safety_factor"]
        self.u_last = None

    # ----- design description ----------------------------------------------------------
    def density(self, x):
        rho = np.asarray(self.Hf @ x).ravel() / self.Hs
        return np.where(self.solid, 1.0, np.where(self.void, 0.0, rho))

    def to_design(self, d_rho):
        """Chain rule from physical densities back to design variables."""
        return np.asarray(self.Hf.T @ (np.where(self.active, d_rho, 0.0) / self.Hs)).ravel()

    # ----- analysis ----------------------------------------------------------------------
    def solve(self, rho):
        m = self.m
        for u0, steps in ((self.u_last, 1), (None, 1), (None, 4), (None, 16)):
            try:
                u, info = m.solve(rho, self.F, u0=u0, steps=steps)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("Newton failed even with 16 load steps")
        self.u_last = u
        return u, info

    def evaluate(self, x, gradients=True, rho=None):
        """Objective (negative closing displacement in mm), constraints g <= 0, and gradients in x.
        Pass rho to analyse a given density field directly (no gradients)."""
        c, m = self.cfg, self.m
        rho = self.density(x) if rho is None else rho
        u, info = self.solve(rho)
        closing, stroke = self.out_weights @ u, self.in_weights @ u
        sig, dsig = m.center_stress(u, rho)
        vm, dvm = von_mises(sig)
        q = c["stress_exponent"]
        ratio = np.where(self.active, rho**q * vm / self.sigma_allow, 0.0)
        value, w = ks(ratio, c["ks_rho"])
        act = self.active.sum()
        out = {"u": u, "rho": rho, "newton": info["history"], "closing": float(closing), "stroke": float(stroke),
               "volume_fraction": float(rho[self.active].mean()),
               "mass": 2 * c["rho"] * m.a**2 * m.t * float(rho.sum()),
               "stress_ratio": ratio, "stress_max": float(ratio.max() * self.sigma_allow),
               "f": -closing / U_REF,
               "g_volume": rho[self.active].mean() / c["volume_fraction"] - 1.0,
               "g_stroke": stroke / c["stroke_max"] - 1.0, "g_stress": value - 1.0}
        if gradients:
            dg_stress = np.zeros(m.ndof)
            np.add.at(dg_stress, m.edof,
                      np.einsum("e,ei,eib->eb", w * np.where(self.active, rho**q, 0.0) / self.sigma_allow, dvm, dsig))
            adj = lambda dg: m.adjoint_gradient(u, rho, info["lu"], dg)
            out["df"] = self.to_design(adj(-self.out_weights / U_REF))
            out["dg_stroke"] = self.to_design(adj(self.in_weights / c["stroke_max"]))
            explicit = w * q * np.maximum(rho, 1e-12) ** (q - 1) * vm / self.sigma_allow
            out["dg_stress"] = self.to_design(adj(dg_stress) + explicit)
            out["dg_volume"] = self.to_design(np.full(m.ne, 1.0 / (act * c["volume_fraction"])))
        return out
