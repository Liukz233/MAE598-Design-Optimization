"""Verification ladder for the plate solver and its adjoint gradients.

Each test returns rows of (id, test, quantity, computed, exact, relative error).
The exact values are closed-form results derived in Section 4 of the report.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from plate import SIGMA, Plate
from problem import CONFIG, HeatSpreader

K, T_INF = CONFIG["k"], CONFIG["T_inf"]


def row(vid, test, quantity, computed, exact, unit="", tol=1e-2, scale=None):
    """One ladder entry. The error is relative to |exact|, or to `scale` when exact is 0."""
    err = abs(computed - exact) / (abs(exact) if scale is None else scale)
    return {"id": vid, "test": test, "quantity": quantity, "computed": float(computed),
            "exact": float(exact), "rel_error": float(err), "unit": unit, "pass": bool(err < tol)}


def edge_load(p: Plate, q_line: float) -> np.ndarray:
    """Heat flow q_line (W/m) entering through the edge x = 0."""
    Q = np.zeros(p.nn)
    left = np.flatnonzero(p.xy[:, 0] < 1e-12)
    Q[left] = q_line * p.dy
    Q[left[[0, -1]]] *= 0.5
    return Q


def v1_linear_fin():
    """Straight fin heated at x = 0, insulated tip, convection on one face."""
    L, W, t, h, q_line = 0.20, 0.02, 2.0e-3, 10.0, 250.0
    p = Plate(L, W, 160, 4, K, h, 0.0, T_INF)
    T, _ = p.solve(np.full(p.ne, t), edge_load(p, q_line))
    m = np.sqrt(h / (K * t))
    base = q_line / (K * t * m * np.tanh(m * L))
    tip = q_line / (K * t * m * np.sinh(m * L))
    return [row("V1", "Linear fin, exact solution", "base temperature rise", T[0] - T_INF, base, "K"),
            row("V1", "Linear fin, exact solution", "tip temperature rise", T[p.nx] - T_INF, tip, "K")]


def v2_manufactured():
    """Manufactured solution of the full nonlinear equation with varying thickness."""
    Lx, Ly, t0, theta0 = 0.30, 0.20, 3.0e-3, 40.0
    h, eps = CONFIG["h"], CONFIG["eps"]

    def theta(x, y):
        return theta0 * np.cos(np.pi * x / Lx) * np.cos(np.pi * y / Ly)

    def source(x, y):
        t = t0 * (1 + x / Lx)
        th = theta(x, y)
        dth_dx = -theta0 * np.pi / Lx * np.sin(np.pi * x / Lx) * np.cos(np.pi * y / Ly)
        conduction = K * t * np.pi**2 * (1 / Lx**2 + 1 / Ly**2) * th - K * (t0 / Lx) * dth_dx
        return conduction + h * th + eps * SIGMA * ((T_INF + th) ** 4 - T_INF**4)

    sizes, errors = [], []
    for n in (6, 12, 24, 48, 96):
        p = Plate(Lx, Ly, n, 2 * n // 3, K, h, eps, T_INF)
        t_el = t0 * (1 + p.centers[:, 0] / Lx)
        T, _ = p.solve(t_el, p.load_function(source))
        e = T - T_INF - theta(p.xy[:, 0], p.xy[:, 1])
        errors.append(np.sqrt(e @ (p.M @ e) / (Lx * Ly)))  # L2 norm of the nodal error
        sizes.append(p.dx)
    slope = np.polyfit(np.log(sizes[1:]), np.log(errors[1:]), 1)[0]
    rows = [row("V2", "Manufactured solution, mesh refinement", "L2 error convergence rate",
                slope, 2.0, "", tol=0.05)]
    return rows, {"h_mm": [1e3 * s for s in sizes], "l2_error_K": errors, "slope": float(slope)}


def v3_heat_balance():
    """Heat leaving the surface equals heat generated, on the design model."""
    rows = []
    hs = HeatSpreader()
    for label, x in (("uniform 2 mm", np.full(hs.n, 1 / 7)), ("random thickness", None)):
        if x is None:
            x = np.random.default_rng(3).uniform(0, 1, hs.n)
        hs.T_last = None
        T, _ = hs.solve(x)
        rows.append(row("V3", "Global heat balance", f"heat leaving the plate, {label}",
                        hs.plate.heat_out(T), hs.Q.sum(), "W", tol=1e-8))
    return rows


def v4_invariance():
    """Zero modes of the conduction matrix and invariance under mirroring and rotation."""
    hs = HeatSpreader()
    x = np.random.default_rng(4).uniform(0, 1, hs.n)
    p = hs.plate
    t_el = hs.thickness(x)[hs.tile_of_element]
    lam = np.sort(np.linalg.eigvalsh(p.conduction(t_el).toarray()))
    n_zero = int(np.sum(lam < 1e-9 * lam[-1]))
    hs.T_last = None
    T, _ = hs.solve(x)

    # mirror the plate in x: thickness and load are flipped, the field must flip with them
    nx, ny = p.nx, p.ny
    T_m, _ = p.solve(t_el.reshape(ny, nx)[:, ::-1].ravel(),
                     hs.Q.reshape(ny + 1, nx + 1)[:, ::-1].ravel())
    mirror = np.abs(T_m.reshape(ny + 1, nx + 1)[:, ::-1].ravel() - T).max()

    # rotate by 90 degrees: swap the x and y axes of the mesh
    pr = Plate(p.Ly, p.Lx, ny, nx, p.k, p.h, p.eps, p.T_inf)
    T_r, _ = pr.solve(t_el.reshape(ny, nx).T.ravel(), hs.Q.reshape(ny + 1, nx + 1).T.ravel())
    rotate = np.abs(T_r.reshape(nx + 1, ny + 1).T.ravel() - T).max()
    rise = (T - p.T_inf).max()
    return [row("V4", "Zero modes of the conduction matrix", "number of zero eigenvalues",
                n_zero, 1, "", tol=1e-12),
            row("V4", "Mirror the plate in x", "largest temperature difference",
                mirror, 0.0, "K", tol=1e-9, scale=rise),
            row("V4", "Rotate the plate by 90 degrees", "largest temperature difference",
                rotate, 0.0, "K", tol=1e-9, scale=rise)]


def v5_nonlinear():
    """Radiation benchmarks with closed-form answers, and the Newton convergence order."""
    eps = CONFIG["eps"]
    rows = []

    # (a) uniformly heated plate, radiation only: eps*sigma*(T^4 - T_inf^4) = q
    q = 400.0
    p = Plate(0.30, 0.20, 12, 8, K, 0.0, eps, T_INF)
    T, _ = p.solve(np.full(p.ne, 2e-3), p.load_uniform(np.full(p.ne, q)))
    exact = (T_INF**4 + q / (eps * SIGMA)) ** 0.25
    rows.append(row("V5", "Uniformly heated radiating plate", "plate temperature", T.mean(), exact, "K",
                    tol=1e-8))

    # (b) radiating fin, insulated tip. First integral of k t T'' = eps*sigma*(T^4 - T_inf^4):
    #     q_line^2 = 2 k t eps sigma [ (T0^5 - TL^5)/5 - T_inf^4 (T0 - TL) ]
    L, W, t, q_line = 0.20, 0.02, 2.0e-3, 250.0
    p = Plate(L, W, 320, 4, K, 0.0, eps, T_INF)
    T, info = p.solve(np.full(p.ne, t), edge_load(p, q_line))
    T0, TL = T[0], T[p.nx]
    q_from_T = np.sqrt(2 * K * t * eps * SIGMA * ((T0**5 - TL**5) / 5 - T_INF**4 * (T0 - TL)))
    rows.append(row("V5", "Radiating fin, energy first integral", "base heat flow", q_from_T, q_line, "W/m"))

    # (c) Newton convergence on the design model
    hs = HeatSpreader()
    _, info = hs.solve(np.full(hs.n, 1 / 7))
    r = np.array(info["history"])
    order = np.log(r[3] / r[2]) / np.log(r[2] / r[1])
    rows.append(row("V5", "Newton's method on the design model", "observed convergence order", order, 2.0,
                    "", tol=0.15))

    # (d) the residual is the gradient of a potential (the energy form used in Projects 2 and 3)
    p = hs.plate
    rng = np.random.default_rng(5)
    t_el = hs.thickness(rng.uniform(0, 1, hs.n))[hs.tile_of_element]
    T = p.T_inf + rng.uniform(5, 50, p.nn)
    v = rng.normal(size=p.nn)
    step = 1e-4
    dPi = (p.potential(T + step * v, t_el, hs.Q) - p.potential(T - step * v, t_el, hs.Q)) / (2 * step)
    rows.append(row("V5", "Residual as the gradient of the potential", "directional derivative of the potential",
                    p.residual(T, t_el, hs.Q) @ v, dPi, "", tol=1e-6))
    return rows, {"residuals": r.tolist(), "order": float(order)}


def v6_gradients(x=None):
    """Adjoint directional derivatives against central differences (Taylor test)."""
    rows, sweeps = [], {}
    steps = 10.0 ** -np.arange(1, 9)
    rng = np.random.default_rng(6)
    for model in ("full", "linearized"):
        hs = HeatSpreader(radiation=model)
        x0 = rng.uniform(0.1, 0.9, hs.n) if x is None else x
        base = hs.evaluate(x0)
        for name in ("chip", "optic"):
            worst = (-1.0, None, None)  # (error, adjoint, finite difference)
            for trial in range(3):
                v = rng.normal(size=hs.n)
                ad = base["dg_" + name] @ v
                fd = []
                for s in steps:
                    hs.T_last = None
                    gp = hs.evaluate(x0 + s * v, gradients=False)["g_" + name]
                    hs.T_last = None
                    gm = hs.evaluate(x0 - s * v, gradients=False)["g_" + name]
                    fd.append((gp - gm) / (2 * s))
                errs = np.abs(np.array(fd) - ad) / abs(ad)
                if trial == 0:
                    sweeps[f"{model}/{name}"] = errs.tolist()
                i = int(np.argmin(errs))
                if errs[i] > worst[0]:
                    worst = (errs[i], ad, fd[i])
            best_ad, best_fd = worst[1], worst[2]
            label = "nonlinear model" if model == "full" else "linearized model"
            rows.append(row("V6", f"Adjoint vs central difference, {label}",
                            f"directional derivative of g_{name} (worst of 3 directions)",
                            best_ad, best_fd, "", tol=1e-5))
    # self-adjoint special case (Project 1): for thermal compliance C = Q.theta in the linear
    # model the adjoint is -theta, so dC/dt_e = -k theta_e^T k0 theta_e without an adjoint solve
    hs = HeatSpreader(radiation="linearized")
    p = hs.plate
    T, info = hs.solve(rng.uniform(0, 1, hs.n))
    general = p.adjoint_gradient(T, info["lu"], hs.Q)
    theta = (T - p.T_inf)[p.edof]
    closed = -p.k * np.einsum("ei,ij,ej->e", theta, p.k0, theta)
    rows.append(row("V6", "Self-adjoint case, Project 1 formula",
                    "largest difference in the compliance sensitivity",
                    np.abs(general - closed).max(), 0.0, "", tol=1e-9, scale=np.abs(closed).max()))
    slope = np.polyfit(np.log(steps[:3]), np.log(sweeps["full/chip"][:3]), 1)[0]
    return rows, {"steps": steps.tolist(), "errors": sweeps, "slope": float(slope)}


def run_ladder():
    rows = v1_linear_fin()
    r2, mms = v2_manufactured()
    rows += r2 + v3_heat_balance() + v4_invariance()
    r5, newton = v5_nonlinear()
    r6, taylor = v6_gradients()
    rows += r5 + r6
    return rows, {"mms": mms, "newton": newton, "taylor": taylor}


if __name__ == "__main__":
    rows, extra = run_ladder()
    for r in rows:
        print(f"{r['id']} {'ok ' if r['pass'] else 'FAIL'} {r['test']:<46s} {r['quantity']:<58s} "
              f"{r['computed']:.9g}  {r['exact']:.9g}  {r['rel_error']:.2e}")
    print("MMS", extra["mms"])
    print("Newton", extra["newton"])
    print("Taylor slope", extra["taylor"]["slope"], extra["taylor"]["errors"]["full/chip"])
