"""Run the whole Project 4 study and regenerate the report site.

    python projects/04_engineering_design/src/run_all.py

Writes site/assets/results.json, then calls make_site.py, which draws the
figures and fills report/report.md and site/index.html from
report/report.template.md. Every number on the site comes from results.json.
"""

from __future__ import annotations

import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mma import MMA  # noqa: E402
from problem import CONFIG, U_REF, Gripper  # noqa: E402
from verify import run_ladder  # noqa: E402

SITE = Path(__file__).resolve().parents[1] / "site"
NAMES = ("volume", "stroke", "stress")


def optimize(nonlinear, x0=None, cfg=CONFIG, iterations=300, move=0.2, tol=0.005):
    """Maximize the jaw closing with MMA. Objective in mm; constraints written as g <= 0."""
    g = Gripper(cfg, nonlinear)
    x = np.full(g.n, cfg["volume_fraction"]) if x0 is None else x0.copy()
    mma = MMA(g.n, len(NAMES), move=move)
    path, t0 = [], time.perf_counter()
    for it in range(iterations):
        e = g.evaluate(x)
        lam_x, lam = mma.step(x, e["f"], e["df"], np.array([e["g_" + c] for c in NAMES]),
                              np.array([e["dg_" + c] for c in NAMES]))
        path.append({"x": x.copy(), "closing": e["closing"], "stroke": e["stroke"], "stress": e["stress_max"],
                     "volume": e["volume_fraction"]})
        change = np.abs(lam_x - x).max()
        if change < tol and it > 30:
            break
        x = lam_x
    e = g.evaluate(x)
    return {"gripper": g, "x": x, "e": e, "multipliers": dict(zip(NAMES, lam)), "path": path,
            "iterations": len(path), "seconds": time.perf_counter() - t0}


def metrics(x, cfg=CONFIG, nonlinear=True, rho=None, nelx=None, nely=None):
    """Objective and constraints of a design, evaluated with the given kinematic model."""
    g = Gripper(cfg, nonlinear, nelx, nely)
    e = g.evaluate(x, gradients=False, rho=rho)
    r = e["rho"][g.active]
    out = {"closing_mm": 1e3 * e["closing"], "stroke_mm": 1e3 * e["stroke"], "stress_MPa": e["stress_max"] / 1e6,
           "volume_fraction": e["volume_fraction"], "mass_g": 1e3 * e["mass"],
           "gray_fraction": float(np.mean((r > 0.1) & (r < 0.9))),
           "volume_percent": 100 * e["volume_fraction"], "gray_percent": 100 * float(np.mean((r > 0.1) & (r < 0.9))),
           "grip_force_N": cfg["k_out"] * e["closing"], "amplification": e["closing"] / e["stroke"],
           "g_volume": float(e["g_volume"]), "g_stroke": float(e["g_stroke"]), "g_stress": float(e["g_stress"]),
           "newton_iterations": len(e["newton"]) - 1}
    out["pass_stroke"] = bool(out["stroke_mm"] <= 1e3 * cfg["stroke_max"] * 1.001)
    out["pass_stress"] = bool(out["stress_MPa"] <= g.sigma_allow / 1e6 * 1.001)
    out["pass_volume"] = bool(out["volume_fraction"] <= cfg["volume_fraction"] * 1.001)
    out["feasible"] = out["pass_stroke"] and out["pass_stress"] and out["pass_volume"]
    return out, g, e


def grid(g, element=None, nodal=None):
    """Reorder top88 arrays (column by column, top first) to rows from the centre line upwards."""
    m = g.m
    if element is not None:
        return np.asarray(element).reshape(m.nelx, m.nely)[:, ::-1].T.ravel()
    return np.asarray(nodal).reshape(m.nelx + 1, m.nely + 1)[:, ::-1].T.ravel()


def load_path(g, rho, factors):
    """Displacements as the actuator force is raised."""
    u, out = None, []
    for f in factors:
        u = np.zeros(g.m.ndof) if f == 0 else g.m.solve(rho, f * g.F, u0=u, steps=1 if u is not None else 4)[0]
        out.append(u)
    return out


def design_record(label, x, cfg=CONFIG):
    """Everything the 3D viewer needs for one design (large-displacement model)."""
    m, g, e = metrics(x, cfg)
    path = load_path(g, e["rho"], np.linspace(0, 1, 6))
    return {"label": label, "metrics": m,
            "rho": np.round(grid(g, e["rho"]), 3).tolist(),
            "utilization": np.round(grid(g, e["stress_ratio"]), 3).tolist(),
            "path": [{"ux_mm": np.round(1e3 * grid(g, nodal=u[0::2]), 4).tolist(),
                      "uy_mm": np.round(1e3 * grid(g, nodal=u[1::2]), 4).tolist()} for u in path]}


def kkt_report(run, cfg=CONFIG):
    """Stationarity of the Lagrangian at the optimum, with the multipliers of the last MMA step."""
    e, x, lam = run["e"], run["x"], run["multipliers"]
    r = e["df"] + sum(lam[c] * e["dg_" + c] for c in NAMES)
    lower, upper = x < 1e-3, x > 1 - 1e-3
    free = ~(lower | upper)
    scale = np.abs(e["df"]).max()
    g = run["gripper"]
    return {"n_free": int(free.sum()), "n_lower": int(lower.sum()), "n_upper": int(upper.sum()),
            "stationarity_free_percent": 100 * float(np.abs(r[free]).max() / scale),
            "multipliers": {c: float(v) for c, v in lam.items()},
            # d(closing*)/d(limit) = lambda * U_REF / limit, because g = response / limit - 1
            "closing_per_mm_stroke": float(lam["stroke"] * U_REF / cfg["stroke_max"]),
            "closing_mm_per_MPa": float(1e3 * lam["stress"] * U_REF / g.sigma_allow * 1e6),
            "closing_mm_per_percent_volume": float(1e3 * lam["volume"] * U_REF / cfg["volume_fraction"] * 0.01)}


def gradient_cost(x, sample=40):
    """Time one adjoint gradient evaluation against forward differences on a sample of variables."""
    g = Gripper()
    g.evaluate(x)
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        g.u_last = None
        e = g.evaluate(x)
    t_adjoint = (time.perf_counter() - t0) / reps
    rng = np.random.default_rng(7)
    picks = rng.choice(np.flatnonzero(g.active), sample, replace=False)
    step = 1e-6
    t0 = time.perf_counter()
    g.u_last = None
    base = g.evaluate(x, gradients=False)
    worst = 0.0
    for j in picks:
        xp = x.copy()
        xp[j] += step
        g.u_last = None
        ep = g.evaluate(xp, gradients=False)
        for key, dkey in (("f", "df"), ("g_stroke", "dg_stroke"), ("g_stress", "dg_stress")):
            worst = max(worst, abs((ep[key] - base[key]) / step - e[dkey][j]) / np.abs(e[dkey]).max())
    per_solve = (time.perf_counter() - t0) / (sample + 1)
    return {"adjoint_seconds": t_adjoint, "fd_seconds_per_solve": per_solve, "fd_sample": sample,
            "fd_seconds": per_solve * (g.n + 1), "fd_minutes": per_solve * (g.n + 1) / 60, "speedup": per_solve * (g.n + 1) / t_adjoint,
            "n_variables": g.n, "n_dof": int(g.m.free.size), "fd_solves": g.n + 1, "max_difference": worst}


def main():
    t_begin = time.perf_counter()
    cfg = CONFIG

    print("optimization with linear analysis ...")
    lin = optimize(False)
    print(f"  {lin['iterations']} iterations, {lin['seconds']:.0f} s")
    print("optimization with large-displacement analysis, starting from the linear optimum ...")
    opt = optimize(True, lin["x"], move=0.1)
    print(f"  {opt['iterations']} iterations, {opt['seconds']:.0f} s")
    print("other starts ...")
    uniform = optimize(True)
    lin_half = optimize(False, np.full(lin["x"].size, 0.5))
    x_opt, x_lin = opt["x"], lin["x"]
    starts = []
    for name, run in (("Linear optimum, then large displacement (reported design)", opt),
                      ("Uniform density 0.3, large displacement throughout", uniform),
                      ("Uniform density 0.3, linear analysis only", lin),
                      ("Uniform density 0.5, linear analysis only", lin_half)):
        m = metrics(run["x"])[0]
        starts.append({"start": name, "iterations": run["iterations"], "seconds": run["seconds"], **m})
        print(f"  {name}: closing {m['closing_mm']:.3f} mm, feasible {m['feasible']}")

    print("verification ladder ...")
    ladder, extra = run_ladder(x_opt)
    for r in ladder:
        print(f"  {r['id']} {'ok  ' if r['pass'] else 'FAIL'} {r['test']}: {r['quantity']}  "
              f"{r['computed']:.8g} vs {r['exact']:.8g}  ({r['rel_error']:.1e})")

    print("multipliers, mesh check, gradient cost ...")
    opt_metrics, g_opt, e_opt = metrics(x_opt)
    kkt = kkt_report(opt)
    relaxed = copy.deepcopy(cfg)
    relaxed["stroke_max"] *= 1.02
    again = optimize(True, x_opt, cfg=relaxed, iterations=80, move=0.05)
    kkt["stroke_relaxation_mm"] = 1e3 * (relaxed["stroke_max"] - cfg["stroke_max"])
    kkt["reoptimized_closing_per_mm_stroke"] = (metrics(again["x"], relaxed)[0]["closing_mm"]
                                                - opt_metrics["closing_mm"]) / kkt["stroke_relaxation_mm"]

    models = {"nonlinear": {"own_model": opt_metrics, "nonlinear_model": opt_metrics,
                            "linear_model": metrics(x_opt, nonlinear=False)[0]},
              "linear": {"own_model": metrics(x_lin, nonlinear=False)[0], "nonlinear_model": metrics(x_lin)[0]}}
    lin_own, lin_true = models["linear"]["own_model"], models["linear"]["nonlinear_model"]
    nonlinear_summary = {
        "closing_overprediction_percent": 100 * (lin_own["closing_mm"] / lin_true["closing_mm"] - 1),
        "stroke_unused_mm": 1e3 * cfg["stroke_max"] - lin_true["stroke_mm"],
        "closing_gain_percent": 100 * (opt_metrics["closing_mm"] / lin_true["closing_mm"] - 1),
        "model_gap_reported_percent": 100 * (models["nonlinear"]["linear_model"]["closing_mm"]
                                             / opt_metrics["closing_mm"] - 1),
        "material_saving_percent": 100 * (1 - opt_metrics["volume_fraction"] / lin_true["volume_fraction"]),
    }

    # the same density field on a mesh twice as fine
    mesh_check = [{"element_mm": 1e3 * g_opt.m.a, "unknowns": int(g_opt.m.free.size), **{
        k: opt_metrics[k] for k in ("closing_mm", "stroke_mm", "stress_MPa")}}]
    m0 = g_opt.m
    fine = np.repeat(np.repeat(e_opt["rho"].reshape(m0.nelx, m0.nely), 2, axis=0), 2, axis=1).ravel()
    mf, gf, _ = metrics(None, rho=fine, nelx=2 * m0.nelx, nely=2 * m0.nely)
    mesh_check.append({"element_mm": 1e3 * gf.m.a, "unknowns": int(gf.m.free.size),
                       **{k: mf[k] for k in ("closing_mm", "stroke_mm", "stress_MPa")}})

    cost = gradient_cost(x_opt)

    # force-closing paths of the two optima, each with both kinematic models
    factors = np.linspace(0, 1.2, 13)
    path = {"force_N": (factors * cfg["F_in"]).tolist()}
    for name, x in (("optimized", x_opt), ("linear_optimum", x_lin)):
        path[name] = {}
        for nonlinear in (True, False):
            g = Gripper(nonlinear=nonlinear)
            us = load_path(g, g.density(x), factors)
            path[name]["nonlinear" if nonlinear else "linear"] = {
                "closing_mm": [1e3 * float(g.out_weights @ u) for u in us],
                "stroke_mm": [1e3 * float(g.in_weights @ u) for u in us]}

    # optimization history of the large-displacement run, with its linear first stage
    hist = {"iteration": [], "closing_mm": [], "stroke_mm": [], "stress_MPa": [], "volume_fraction": [],
            "stage_change": len(lin["path"])}
    both = lin["path"] + opt["path"]
    for i, p in enumerate(both):
        hist["iteration"].append(i)
        hist["closing_mm"].append(round(1e3 * p["closing"], 4))
        hist["stroke_mm"].append(round(1e3 * p["stroke"], 4))
        hist["stress_MPa"].append(round(p["stress"] / 1e6, 3))
        hist["volume_fraction"].append(round(p["volume"], 4))
    keep = np.unique(np.round(np.linspace(0, np.sqrt(len(both) - 1), 32) ** 2).astype(int))
    frames = [{"iteration": int(i), "rho": np.round(grid(g_opt, g_opt.density(both[i]["x"])), 2).tolist(),
               "metrics": {"closing_mm": hist["closing_mm"][i], "stroke_mm": hist["stroke_mm"][i],
                           "stress_MPa": hist["stress_MPa"][i], "volume_fraction": hist["volume_fraction"][i]}}
              for i in keep]

    m = g_opt.m
    results = {
        "meta": {"command": "python projects/04_engineering_design/src/run_all.py",
                 "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "config": {"length_mm": 1e3 * cfg["length"], "half_height_mm": 1e3 * cfg["half_height"],
                   "height_mm": 2e3 * cfg["half_height"], "thickness_mm": 1e3 * cfg["thickness"],
                   "nx": m.nelx, "ny": m.nely, "element_mm": 1e3 * m.a, "n_unknowns": int(m.free.size),
                   "n_variables": g_opt.n, "n_active": int(g_opt.active.sum()),
                   "gap_length_mm": 1e3 * cfg["gap_length"], "gap_half_height_mm": 1e3 * cfg["gap_half_height"],
                   "gap_mm": 2e3 * cfg["gap_half_height"], "support_height_mm": 1e3 * cfg["support_height"],
                   "pad_mm": 1e3 * cfg["pad"], "E_GPa": cfg["E"] / 1e9, "nu": cfg["nu"], "rho": cfg["rho"],
                   "S_y_MPa": cfg["S_y"] / 1e6, "safety_factor": cfg["safety_factor"],
                   "sigma_allow_MPa": g_opt.sigma_allow / 1e6, "F_in_N": cfg["F_in"],
                   "stroke_max_mm": 1e3 * cfg["stroke_max"], "k_out_N_per_mm": cfg["k_out"] / 1e3,
                   "volume_percent": 100 * cfg["volume_fraction"],
                   "penal": cfg["penal"], "E_min": cfg["E_min"], "rmin": cfg["rmin"],
                   "rmin_mm": cfg["rmin"] * 1e3 * m.a, "volume_fraction": cfg["volume_fraction"],
                   "stress_exponent": cfg["stress_exponent"], "ks_rho": cfg["ks_rho"],
                   "ks_bound_percent": 100 * np.log(g_opt.active.sum()) / cfg["ks_rho"],
                   "rho_switch": m.rho_switch, "stiffness_at_switch_percent": 100 * m.rho_switch ** cfg["penal"],
                   "strain_allow_percent": 100 * g_opt.sigma_allow / cfg["E"]},
        "designs": {"initial": design_record("Initial: uniform density 0.3", np.full(g_opt.n, cfg["volume_fraction"])),
                    "optimized": design_record("Optimized, large-displacement analysis", x_opt),
                    "linear": design_record("Optimized with linear analysis", x_lin)},
        "optimization": {"iterations": opt["iterations"], "seconds": opt["seconds"],
                         "linear_iterations": lin["iterations"], "linear_seconds": lin["seconds"],
                         "starts": starts, "kkt": kkt, "mesh_check": mesh_check,
                         "finite_strain_elements": int(g_opt.m.finite(e_opt["rho"]).sum())},
        "models": models, "nonlinear": nonlinear_summary, "load_path": path,
        "history": hist, "frames": frames,
        "verification": {"ladder": ladder, **extra, "n_checks": len(ladder),
                         "all_pass": all(r["pass"] for r in ladder)},
        "gradient_cost": cost,
    }
    p1_row = next(r for r in ladder if r["test"].startswith("Project 1"))
    results["verification"]["p1_digits"] = int(-np.log10(max(p1_row["rel_error"], 1e-16)))
    res = extra["newton"]["residuals"]
    results["verification"]["newton"]["residuals_text"] = ", ".join(f"{v:.1e}" for v in res)
    results["verification"]["newton"]["iterations_from_rest"] = len(extra["newton"]["from_rest"]) - 1
    elastica = [r for r in ladder if r["test"].startswith("Cantilever")]
    results["verification"]["elastica_error_percent"] = 100 * max(r["rel_error"] for r in elastica)
    results["optimization"]["mesh_change_percent"] = {
        k: 100 * (mesh_check[1][k] / mesh_check[0][k] - 1) for k in ("closing_mm", "stroke_mm", "stress_MPa")}
    results["meta"]["run_minutes"] = (time.perf_counter() - t_begin) / 60

    out = SITE / "assets" / "results.json"
    out.write_text(json.dumps(results, separators=(",", ":"), default=float))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) in {60 * results['meta']['run_minutes']:.0f} s")

    import make_site
    make_site.main()


if __name__ == "__main__":
    main()
