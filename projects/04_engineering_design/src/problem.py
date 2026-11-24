"""Design problem: lightest stainless-steel plate beam on the Project 1 MBB layout.

The half beam is divided into square tiles. Design variable x_j in [0, 1] sets
the plate thickness of tile j, t_j = t_min + (t_max - t_min) x_j. SI units.
"""

from __future__ import annotations

import numpy as np

from beam import Beam, ks, von_mises

CONFIG = {
    # geometry: half of a simply supported beam, symmetric about midspan (Project 1 layout, 3:1)
    "half_span": 0.600, "depth": 0.200,     # m
    "tiles_x": 30, "tiles_y": 10,          # 20 mm tiles, one design variable each
    "refine": 2,                            # finite elements per tile edge
    "t_min": 3.0e-3, "t_max": 20.0e-3,     # m
    # austenitic stainless steel 1.4301 (AISI 304)
    "E": 200e9, "nu": 0.3, "rho": 7900.0,
    "f_y": 210e6, "n": 6.0,                # Ramberg-Osgood: eps = s/E + 0.002 (s/f_y)^n
    # loading: one wheel load at midspan, spread over bearing pads
    "P": 40e3,                              # N, whole beam
    "pad": 0.040,                           # m, pad length on the half model
    # requirements
    "sigma_allow": 210e6 / 1.1,            # f_y / gamma_M0
    "deflection_limit": 1.200 / 600,       # span / 600
    "ks_rho": 200.0,
}


class BeamDesign:
    """Analysis, responses and adjoint gradients for one material model."""

    names = ("stress", "deflection")

    def __init__(self, cfg=CONFIG, nonlinear=True, refine=None):
        self.cfg, self.nonlinear = cfg, nonlinear
        rf = cfg["refine"] if refine is None else refine
        self.tx, self.ty, self.rf = cfg["tiles_x"], cfg["tiles_y"], rf
        nelx, nely = self.tx * rf, self.ty * rf
        a = cfg["half_span"] / nelx
        self.beam = b = Beam(nelx, nely, a, cfg["E"], cfg["nu"], cfg["f_y"], cfg["n"], nonlinear)
        self.n = self.tx * self.ty
        ex, ey = np.divmod(np.arange(b.ne), nely)           # top88 element numbering
        self.tile_of_element = (ex // rf) * self.ty + ey // rf
        self.tile_area = (cfg["half_span"] / self.tx) ** 2
        self.F = self.pad_loads(cfg["P"])
        self.mid_dof = 2 * nely + 1                         # vertical displacement, midspan soffit
        self.u_last = None

    def pad_loads(self, P):
        """Half the load pushes down on the top pad at midspan; the support pad pushes up.
        The two are in equilibrium, so the roller of Project 1 only removes the rigid motion."""
        b, cfg = self.beam, self.cfg
        F = np.zeros(b.ndof)
        line = 0.5 * P / cfg["pad"]                          # N/m along the pad
        x, y = b.xy.T
        top = np.flatnonzero((np.abs(y - cfg["depth"]) < 1e-9) & (x <= cfg["pad"] + 1e-9))
        bottom = np.flatnonzero((np.abs(y) < 1e-9) & (x >= cfg["half_span"] - cfg["pad"] - 1e-9))
        for nodes, sign in ((top, -1.0), (bottom, 1.0)):
            w = np.full(nodes.size, line * b.a)
            w[[np.argmin(x[nodes]), np.argmax(x[nodes])]] *= 0.5
            F[2 * nodes + 1] += sign * w
        return F

    # ----- design description -------------------------------------------------------
    def thickness(self, x):
        c = self.cfg
        return c["t_min"] + (c["t_max"] - c["t_min"]) * np.asarray(x)

    def mass(self, x):
        """Mass of the whole beam (both halves), kg."""
        return 2 * self.cfg["rho"] * self.tile_area * self.thickness(x).sum()

    def mass_gradient(self):
        c = self.cfg
        return np.full(self.n, 2 * c["rho"] * self.tile_area * (c["t_max"] - c["t_min"]))

    # ----- analysis -----------------------------------------------------------------
    def solve(self, x, t_el=None):
        t_el = self.thickness(x)[self.tile_of_element] if t_el is None else t_el
        try:
            u, info = self.beam.solve(t_el, self.F, u0=self.u_last)
        except RuntimeError:                                 # poor warm start: restart from zero
            u, info = self.beam.solve(t_el, self.F)
        self.u_last = u
        return u, info

    def responses(self, u, solid=None):
        """Constraint values g <= 0, the quantities behind them, and dg/du.

        stress:      KS over tiles of sigma_vM / sigma_allow, minus 1, where sigma_vM is
                     the von Mises stress of the stress averaged over the tile
        deflection:  midspan deflection / limit, minus 1
        `solid` (elements) restricts the average to solid elements, for the Project 1 layout."""
        c, b = self.cfg, self.beam
        sig, dsig = b.center_stress(u)
        weight = np.ones(b.ne) if solid is None else solid.astype(float)
        count = np.bincount(self.tile_of_element, weight, self.n)
        used = count > 0
        mean = np.stack([np.bincount(self.tile_of_element, weight * sig[:, i], self.n) for i in range(3)], 1)
        mean[used] /= count[used, None]
        vm, dvm = von_mises(mean)
        ratio = np.where(used, vm / c["sigma_allow"], 0.0)
        value, w = ks(ratio, c["ks_rho"])
        # chain rule: tile aggregate -> tile mean stress -> element stress -> element displacements
        coef = (w / c["sigma_allow"] / np.maximum(count, 1.0))[:, None] * dvm
        dg_stress = np.zeros(b.ndof)
        np.add.at(dg_stress, b.edof, np.einsum("ei,eib->eb", weight[:, None] * coef[self.tile_of_element], dsig))
        dg_defl = np.zeros(b.ndof)
        dg_defl[self.mid_dof] = -1.0 / c["deflection_limit"]
        return {"g_stress": value - 1.0, "g_deflection": -u[self.mid_dof] / c["deflection_limit"] - 1.0,
                "utilization": ratio, "vm_max": float(ratio.max() * c["sigma_allow"]),
                "deflection": float(-u[self.mid_dof]),
                "dg_du": {"stress": dg_stress, "deflection": dg_defl}}

    def evaluate(self, x, gradients=True):
        """Mass, both constraints, and their gradients in x."""
        u, info = self.solve(x)
        out = self.responses(u)
        out.update({"u": u, "newton": info["history"], "mass": self.mass(x)})
        if gradients:
            dt_dx = self.cfg["t_max"] - self.cfg["t_min"]
            for name in self.names:
                dg_dt = self.beam.adjoint_gradient(u, info["lu"], out["dg_du"][name])
                out["dg_" + name] = dt_dx * np.bincount(self.tile_of_element, dg_dt, self.n)
        return out
