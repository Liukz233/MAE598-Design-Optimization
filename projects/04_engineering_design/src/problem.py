"""Design problem: lightest heat-spreader plate that keeps two components cool.

Design variables x_j in [0, 1], one per square tile of the plate. The tile
thickness is t_j = t_min + (t_max - t_min) x_j. All inputs are in SI units.
"""

from __future__ import annotations

import numpy as np

from plate import SIGMA, Plate, ks

CELSIUS = 273.15

CONFIG = {
    # plate and tiling
    "Lx": 0.300, "Ly": 0.200,          # m
    "tiles_x": 24, "tiles_y": 16,      # 12.5 mm tiles, one design variable each
    "refine": 2,                        # finite elements per tile edge
    "t_min": 1.0e-3, "t_max": 8.0e-3,  # m
    # aluminium 6061-T6, black anodized
    "k": 167.0, "rho": 2700.0, "eps": 0.82,
    # surroundings
    "h": 6.0, "T_inf": 40.0 + CELSIUS,
    # components: centre (m), side (m), power (W), temperature limit (K)
    "chip": {"center": (0.075, 0.100), "side": 0.025, "power": 20.0, "T_max": 85.0 + CELSIUS},
    "optic": {"center": (0.2375, 0.100), "side": 0.025, "power": 1.5, "T_max": 62.0 + CELSIUS},
    "ks_rho": 200.0,
}


class HeatSpreader:
    """Analysis, responses, and adjoint gradients for one model variant.

    radiation = "full":        eps*sigma*(T^4 - T_inf^4), nonlinear state equation
    radiation = "linearized":  h_r = 4 eps sigma T_inf^3 added to h, linear
    radiation = "none":        convection only, linear
    """

    def __init__(self, cfg=CONFIG, radiation="full", refine=None):
        self.cfg, self.radiation = cfg, radiation
        rf = cfg["refine"] if refine is None else refine
        self.tx, self.ty, self.rf = cfg["tiles_x"], cfg["tiles_y"], rf
        h, eps = cfg["h"], cfg["eps"]
        if radiation == "linearized":
            h, eps = h + 4 * cfg["eps"] * SIGMA * cfg["T_inf"] ** 3, 0.0
        elif radiation == "none":
            eps = 0.0
        self.plate = p = Plate(cfg["Lx"], cfg["Ly"], self.tx * rf, self.ty * rf,
                               cfg["k"], h, eps, cfg["T_inf"])
        self.n = self.tx * self.ty
        ix = np.minimum((p.centers[:, 0] / cfg["Lx"] * self.tx).astype(int), self.tx - 1)
        iy = np.minimum((p.centers[:, 1] / cfg["Ly"] * self.ty).astype(int), self.ty - 1)
        self.tile_of_element = ix + iy * self.tx
        self.tile_area = cfg["Lx"] * cfg["Ly"] / self.n

        q = np.zeros(p.ne)
        self.nodes, self.elements = {}, {}
        for name in ("chip", "optic"):
            c = cfg[name]
            inside = np.all(np.abs(p.centers - np.array(c["center"])) < c["side"] / 2, axis=1)
            q[inside] = c["power"] / (inside.sum() * p.area)
            self.elements[name] = np.flatnonzero(inside)
            self.nodes[name] = np.unique(p.edof[inside])
        self.Q = p.load_uniform(q)
        self.T_last = None

    # ----- design description -------------------------------------------------
    def thickness(self, x):
        c = self.cfg
        return c["t_min"] + (c["t_max"] - c["t_min"]) * np.asarray(x)

    def mass(self, x):
        return self.cfg["rho"] * self.tile_area * self.thickness(x).sum()

    def mass_gradient(self):
        c = self.cfg
        return np.full(self.n, c["rho"] * self.tile_area * (c["t_max"] - c["t_min"]))

    # ----- analysis -------------------------------------------------------------
    def solve(self, x):
        t_el = self.thickness(x)[self.tile_of_element]
        T, info = self.plate.solve(t_el, self.Q, T0=self.T_last)
        self.T_last = T
        return T, info

    def constraint(self, T, name):
        """g = KS(theta_i / theta_limit) - 1 <= 0 over the component's nodes."""
        c, nodes = self.cfg, self.nodes[name]
        limit = c[name]["T_max"] - c["T_inf"]
        value, w = ks((T[nodes] - c["T_inf"]) / limit, c["ks_rho"])
        dg_dT = np.zeros(self.plate.nn)
        dg_dT[nodes] = w / limit
        return value - 1.0, dg_dT

    def evaluate(self, x, gradients=True):
        """Mass, both temperature constraints, and their gradients in x."""
        T, info = self.solve(x)
        out = {"T": T, "newton": info["history"], "mass": self.mass(x)}
        dt_dx = self.cfg["t_max"] - self.cfg["t_min"]
        for name in ("chip", "optic"):
            g, dg_dT = self.constraint(T, name)
            out["g_" + name] = g
            out["T_" + name] = T[self.nodes[name]].max()
            if gradients:
                dg_dt_el = self.plate.adjoint_gradient(T, info["lu"], dg_dT)
                out["dg_" + name] = dt_dx * np.bincount(self.tile_of_element, dg_dt_el, self.n)
        return out
