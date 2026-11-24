"""Run the whole Project 4 study and regenerate the report site.

    python projects/04_engineering_design/src/run_all.py

Writes site/assets/results.json, then calls make_site.py, which draws the
figures and fills report/report.md and site/index.html from
report/report.template.md. Every number on the site comes from results.json.
"""

from __future__ import annotations

import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import cg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from beam import von_mises  # noqa: E402
from problem import CONFIG, BeamDesign  # noqa: E402
from verify import run_ladder  # noqa: E402

SITE = Path(__file__).resolve().parents[1] / "site"
PROJECTS = Path(__file__).resolve().parents[2]
T_START = 6.0e-3  # uniform starting thickness, m


def x_uniform(t, cfg=CONFIG):
    n = cfg["tiles_x"] * cfg["tiles_y"]
    return np.full(n, (t - cfg["t_min"]) / (cfg["t_max"] - cfg["t_min"]))


def optimize(nonlinear=True, x0=None, cfg=CONFIG):
    """Minimize mass subject to the stress and deflection limits with SLSQP.

    Scaling: x in [0, 1], mass divided by the mass of the full-thickness plate,
    constraints written as g <= 0 with g of order one."""
    d = BeamDesign(cfg, nonlinear)
    x0 = x_uniform(T_START, cfg) if x0 is None else x0
    m_ref = d.mass(np.ones(d.n))
    cache = {}

    def evaluate(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = d.evaluate(x)
        return cache[key]

    cons = [{"type": "ineq",
             "fun": lambda x, c=c: -evaluate(x)["g_" + c],
             "jac": lambda x, c=c: -evaluate(x)["dg_" + c]} for c in d.names]
    path = [x0.copy()]
    t0 = time.perf_counter()
    res = minimize(lambda x: d.mass(x) / m_ref, x0, jac=lambda x: d.mass_gradient() / m_ref,
                   method="SLSQP", bounds=[(0.0, 1.0)] * d.n, constraints=cons,
                   callback=lambda x: path.append(x.copy()),
                   options={"maxiter": 500, "ftol": 1e-9})
    return {"design": d, "x": res.x, "res": res, "path": path, "m_ref": m_ref,
            "seconds": time.perf_counter() - t0}


def metrics(x, cfg=CONFIG, nonlinear=True, refine=None):
    """Objective and constraints of a design, evaluated with the given material model."""
    d = BeamDesign(cfg, nonlinear, refine)
    e = d.evaluate(x, gradients=False)
    out = {"mass_kg": e["mass"], "stress_MPa": e["vm_max"] / 1e6, "deflection_mm": 1e3 * e["deflection"],
           "g_stress": float(e["g_stress"]), "g_deflection": float(e["g_deflection"]),
           "newton_iterations": len(e["newton"]) - 1}
    out["pass_stress"] = bool(out["g_stress"] <= 1e-6)
    out["pass_deflection"] = bool(out["g_deflection"] <= 1e-6)
    out["feasible"] = out["pass_stress"] and out["pass_deflection"]
    return out, d, e


def to_grid(d, tile_values=None, nodal=None):
    """Reorder top88 arrays (column by column, top first) to rows from the bottom."""
    if tile_values is not None:
        return np.asarray(tile_values).reshape(d.tx, d.ty)[:, ::-1].T.ravel()
    b = d.beam
    return np.asarray(nodal).reshape(b.nelx + 1, b.nely + 1)[:, ::-1].T.ravel()


def design_record(label, x, cfg=CONFIG):
    """Everything the 3D viewer needs for one design (nonlinear material)."""
    m, d, e = metrics(x, cfg)
    return {"label": label, "metrics": m,
            "t_mm": np.round(1e3 * to_grid(d, d.thickness(x)), 3).tolist(),
            "utilization": np.round(to_grid(d, e["utilization"]), 4).tolist(),
            "ux_mm": np.round(1e3 * to_grid(d, nodal=e["u"][0::2]), 5).tolist(),
            "uy_mm": np.round(1e3 * to_grid(d, nodal=e["u"][1::2]), 5).tolist()}


def kkt_report(run):
    """Stationarity of the Lagrangian at the optimum, using SLSQP's multipliers."""
    d, x, m_ref = run["design"], run["x"], run["m_ref"]
    e = d.evaluate(x)
    mu = dict(zip(d.names, np.asarray(run["res"].multipliers, float)))
    r = d.mass_gradient() / m_ref + sum(mu[c] * e["dg_" + c] for c in mu)
    lower, upper = x < 1e-6, x > 1 - 1e-6
    free = ~(lower | upper)
    scale = (d.mass_gradient() / m_ref)[0]
    return {"n_free": int(free.sum()), "n_lower": int(lower.sum()), "n_upper": int(upper.sum()),
            "stationarity_free_percent": 100 * float(np.abs(r[free]).max() / scale),
            "min_reduced_gradient_lower": float(r[lower].min() / scale) if lower.any() else 0.0,
            "multipliers": {c: float(v) for c, v in mu.items()},
            # dm*/d(limit) = -mu * m_ref / limit, because g = response / limit - 1
            "saving_g_per_0p1mm": float(1e3 * mu["deflection"] * m_ref / d.cfg["deflection_limit"] * 1e-4),
            "saving_g_per_MPa": float(1e3 * mu["stress"] * m_ref / d.cfg["sigma_allow"] * 1e6)}


def gradient_cost():
    """Time one adjoint gradient evaluation against forward differences."""
    d = BeamDesign()
    x = x_uniform(T_START)
    d.evaluate(x)
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        d.u_last = None
        e = d.evaluate(x)
    t_adjoint = (time.perf_counter() - t0) / reps

    step = 1e-6
    t0 = time.perf_counter()
    d.u_last = None
    base = d.evaluate(x, gradients=False)
    fd = {c: np.zeros(d.n) for c in d.names}
    for j in range(d.n):
        xp = x.copy()
        xp[j] += step
        d.u_last = None
        ep = d.evaluate(xp, gradients=False)
        for c in fd:
            fd[c][j] = (ep["g_" + c] - base["g_" + c]) / step
    t_fd = time.perf_counter() - t0
    diff = max(float(np.abs(fd[c] - e["dg_" + c]).max() / np.abs(e["dg_" + c]).max()) for c in fd)
    return {"adjoint_seconds": t_adjoint, "fd_seconds": t_fd, "speedup": t_fd / t_adjoint,
            "n_variables": d.n, "n_dof": int(d.beam.free.size), "fd_solves": d.n + 1, "max_difference": diff}


def project1_layout(mass_kg):
    """The Project 1 topology as a plate of the same mass, checked against the Project 4 limits."""
    rho = np.loadtxt(PROJECTS / "01_problem_formulation" / "results" / "final_density.csv", delimiter=",")
    solid = rho.T.ravel() > 0.5                               # top88 element order
    out = {"solid_fraction": float(solid.mean())}
    for nonlinear in (True, False):
        d = BeamDesign(nonlinear=nonlinear, refine=4)         # the 120 x 40 mesh of Project 1
        t_solid = mass_kg / (2 * CONFIG["rho"] * d.beam.a**2 * solid.sum())
        u, _ = d.solve(None, t_el=np.where(solid, t_solid, 1e-6 * t_solid))
        r = d.responses(u, solid=solid)
        out["nonlinear" if nonlinear else "linear"] = {
            "stress_MPa": r["vm_max"] / 1e6, "deflection_mm": 1e3 * r["deflection"],
            "tiles_over_limit": int(np.sum(r["utilization"] > 1))}
        if nonlinear:
            out["thickness_mm"] = 1e3 * t_solid
            out["solid"] = solid.reshape(120, 40)[:, ::-1].T.astype(int).ravel().tolist()
            out["tile_utilization"] = np.round(to_grid(d, r["utilization"]), 3).tolist()
    return out


def load_path(x):
    """Midspan deflection of a design as the load is raised, for both material models."""
    factors = np.linspace(0.0, 1.5, 16)
    out = {"factor": factors.tolist()}
    for nonlinear in (True, False):
        d = BeamDesign(nonlinear=nonlinear)
        t_el = d.thickness(x)[d.tile_of_element]
        u, defl = None, []
        for f in factors:
            u = d.beam.solve(t_el, f * d.F, u0=u)[0] if f > 0 else np.zeros(d.beam.ndof)
            defl.append(-1e3 * u[d.mid_dof])
        out["nonlinear_mm" if nonlinear else "linear_mm"] = defl
    return out


def conditioning(x):
    """Condition number of the tangent stiffness at equilibrium, as measured in Project 2."""
    out = {}
    for nonlinear in (False, True):
        d = BeamDesign(nonlinear=nonlinear)
        b = d.beam
        u, _ = d.solve(x)
        K = b.tangent(u, d.thickness(x)[d.tile_of_element])[b.free, :][:, b.free]
        A = K.toarray()
        dg = A.diagonal()
        lam = np.linalg.eigvalsh(A)
        lam_j = np.linalg.eigvalsh(A / np.sqrt(np.outer(dg, dg)))
        count = [0]
        cg(K, d.F[b.free], rtol=1e-6, atol=0.0, M=diags(1 / dg), maxiter=50000,
           callback=lambda _: count.__setitem__(0, count[0] + 1))
        out["nonlinear" if nonlinear else "linear"] = {
            "kappa": float(lam[-1] / lam[0]), "kappa_jacobi": float(lam_j[-1] / lam_j[0]),
            "pcg_iterations": count[0], "unknowns": int(b.free.size)}
    with open(PROJECTS / "02_gradient_descent" / "results" / "conditioning.csv") as f:
        p2 = {r["case"]: r for r in csv.DictReader(f)}["uniform_48x16"]
    out["project2_uniform"] = {"kappa": float(p2["kappa"]), "unknowns": int(p2["free_dofs"]),
                               "mesh": "48 x 16"}
    return out


def main():
    t_begin = time.perf_counter()
    cfg = CONFIG

    print("verification ladder ...")
    ladder, extra = run_ladder()
    for r in ladder:
        print(f"  {r['id']} {'ok  ' if r['pass'] else 'FAIL'} {r['test']}: {r['quantity']}  "
              f"{r['computed']:.8g} vs {r['exact']:.8g}  ({r['rel_error']:.1e})")

    print("optimization, nonlinear material ...")
    runs = {"nonlinear": optimize(True)}
    x_opt = runs["nonlinear"]["x"]
    rng = np.random.default_rng(0)
    start_points = {"uniform 6 mm": None,
                    "uniform 3 mm (lower bound)": x_uniform(cfg["t_min"]),
                    "uniform 20 mm (upper bound)": x_uniform(cfg["t_max"]),
                    "random": rng.uniform(0, 1, x_opt.size)}
    starts = []
    for name, x0 in start_points.items():
        run = runs["nonlinear"] if x0 is None else optimize(True, x0)
        m = metrics(run["x"])[0]
        starts.append({"start": name, "iterations": int(run["res"].nit), "seconds": run["seconds"],
                       "mass_kg": m["mass_kg"], "stress_MPa": m["stress_MPa"], "deflection_mm": m["deflection_mm"],
                       "max_thickness_difference_mm": float(
                           1e3 * (cfg["t_max"] - cfg["t_min"]) * np.abs(run["x"] - x_opt).max())})
        print(f"  start {name}: {m['mass_kg']:.4f} kg in {run['res'].nit} iterations")

    print("optimization, linear material ...")
    runs["linear"] = optimize(False)
    x_lin = runs["linear"]["x"]

    print("multiplier checks ...")
    opt_metrics = metrics(x_opt)[0]
    m_opt = opt_metrics["mass_kg"]
    kkt = kkt_report(runs["nonlinear"])
    kkt["linear_design"] = kkt_report(runs["linear"])
    relaxed = copy.deepcopy(cfg)
    relaxed["deflection_limit"] += 1e-4                       # 0.1 mm more deflection allowed
    kkt["reoptimized_saving_g_per_0p1mm"] = 1e3 * (m_opt - metrics(optimize(True, cfg=relaxed)["x"], relaxed)[0]["mass_kg"])
    kkt["saving_percent_per_0p1mm"] = 100 * kkt["saving_g_per_0p1mm"] / (1e3 * m_opt)

    # thinnest uniform plate that meets both limits, for reference
    lo, hi = cfg["t_min"], cfg["t_max"]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if metrics(x_uniform(mid))[0]["feasible"] else (mid, hi)
    uniform = metrics(x_uniform(hi))[0]
    uniform["thickness_mm"] = 1e3 * hi

    models = {}
    for name, x in (("nonlinear", x_opt), ("linear", x_lin)):
        models[name] = {"iterations": int(runs[name]["res"].nit), "seconds": runs[name]["seconds"],
                        "success": bool(runs[name]["res"].success),
                        "own_model": metrics(x, nonlinear=(name == "nonlinear"))[0],
                        "nonlinear_model": metrics(x)[0]}
    lin_true = models["linear"]["nonlinear_model"]
    f_y, n_ro, E = cfg["f_y"], cfg["n"], cfg["E"]
    secant = lambda s: 1 / (1 + 0.002 * E / s * (s / f_y) ** n_ro)
    nonlinear_summary = {
        "extra_mass_kg": models["nonlinear"]["own_model"]["mass_kg"] - models["linear"]["own_model"]["mass_kg"],
        "extra_mass_percent": 100 * (models["nonlinear"]["own_model"]["mass_kg"]
                                     / models["linear"]["own_model"]["mass_kg"] - 1),
        "linear_design_deflection_excess_percent": 100 * (lin_true["deflection_mm"]
                                                         / (1e3 * cfg["deflection_limit"]) - 1),
        "secant_ratio_at_allowable": secant(cfg["sigma_allow"]),
        "secant_ratio_at_optimum_stress": secant(1e6 * opt_metrics["stress_MPa"]),
        "secant_ratio_at_100MPa": secant(100e6),
    }

    mesh_check = []
    for rf in (1, 2, 4, 8):
        m, d, _ = metrics(x_opt, refine=rf)
        mesh_check.append({"element_mm": 1e3 * d.beam.a, "unknowns": int(d.beam.free.size),
                           "stress_MPa": m["stress_MPa"], "deflection_mm": m["deflection_mm"]})
    # the stress measure used first, at element centres, for the linear-material design (Section 10)
    element_peaks = []
    for rf in (1, 2, 4, 8):
        _, d, e = metrics(x_lin, nonlinear=False, refine=rf)
        element_peaks.append({"element_mm": 1e3 * d.beam.a,
                              "element_peak_MPa": float(von_mises(d.beam.center_stress(e["u"])[0])[0].max() / 1e6),
                              "tile_peak_MPa": e["vm_max"] / 1e6})

    # shape of the optimum: thickness along the bottom row of tiles and in the middle of the web
    t_grid = 1e3 * to_grid(BeamDesign(), BeamDesign().thickness(x_opt)).reshape(cfg["tiles_y"], cfg["tiles_x"])
    thin = np.flatnonzero(t_grid[0] <= 1e3 * cfg["t_min"] + 0.05)   # where the bottom flange first reaches t_min
    shape = {"flange_midspan_bottom_mm": float(t_grid[0, 0]), "flange_midspan_top_mm": float(t_grid[-1, 0]),
             "flange_length_mm": float(thin.min() * 1e3 * cfg["half_span"] / cfg["tiles_x"]),
             "web_mm": float(np.median(t_grid[1:-1])), "web_max_mm": float(t_grid[1:-1].max()),
             "linear_flange_midspan_bottom_mm": float(1e3 * to_grid(BeamDesign(), BeamDesign().thickness(x_lin))[0])}

    print("Project 1 layout, conditioning, gradient cost ...")
    p1 = project1_layout(m_opt)
    cond = conditioning(x_opt)
    cost = gradient_cost()

    # optimization history of the nonlinear run
    path = runs["nonlinear"]["path"]
    hist = {"iteration": [], "mass_kg": [], "stress_MPa": [], "deflection_mm": []}
    records = []
    d = BeamDesign()
    for i, x in enumerate(path):
        e = d.evaluate(x, gradients=False)
        hist["iteration"].append(i)
        hist["mass_kg"].append(round(e["mass"], 5))
        hist["stress_MPa"].append(round(e["vm_max"] / 1e6, 3))
        hist["deflection_mm"].append(round(1e3 * e["deflection"], 4))
        records.append((x, e))
    keep = np.unique(np.round(np.linspace(0, np.sqrt(len(path) - 1), 36) ** 2).astype(int))
    frames = [{"iteration": int(i), "t_mm": np.round(1e3 * to_grid(d, d.thickness(records[i][0])), 3).tolist(),
               "utilization": np.round(to_grid(d, records[i][1]["utilization"]), 4).tolist(),
               "ux_mm": np.round(1e3 * to_grid(d, nodal=records[i][1]["u"][0::2]), 4).tolist(),
               "uy_mm": np.round(1e3 * to_grid(d, nodal=records[i][1]["u"][1::2]), 4).tolist(),
               "metrics": {"mass_kg": hist["mass_kg"][i], "stress_MPa": hist["stress_MPa"][i],
                           "deflection_mm": hist["deflection_mm"][i]}} for i in keep]

    b = d.beam
    results = {
        "meta": {"command": "python projects/04_engineering_design/src/run_all.py",
                 "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "config": {"half_span_mm": 1e3 * cfg["half_span"], "span_mm": 2e3 * cfg["half_span"],
                   "depth_mm": 1e3 * cfg["depth"], "tiles_x": cfg["tiles_x"], "tiles_y": cfg["tiles_y"],
                   "tile_mm": 1e3 * cfg["half_span"] / cfg["tiles_x"], "refine": cfg["refine"],
                   "nx": b.nelx, "ny": b.nely, "element_mm": 1e3 * b.a, "n_unknowns": int(b.free.size),
                   "n_variables": d.n, "t_min_mm": 1e3 * cfg["t_min"], "t_max_mm": 1e3 * cfg["t_max"],
                   "t_start_mm": 1e3 * T_START, "E_GPa": cfg["E"] / 1e9, "nu": cfg["nu"], "rho": cfg["rho"],
                   "f_y_MPa": cfg["f_y"] / 1e6, "n": cfg["n"], "P_kN": cfg["P"] / 1e3,
                   "pad_mm": 1e3 * cfg["pad"], "pad2_mm": 2e3 * cfg["pad"], "sigma_allow_MPa": cfg["sigma_allow"] / 1e6,
                   "deflection_limit_mm": 1e3 * cfg["deflection_limit"], "ks_rho": cfg["ks_rho"],
                   "ks_bound_percent": 100 * np.log(d.n) / cfg["ks_rho"],
                   "mass_full_plate_kg": runs["nonlinear"]["m_ref"]},
        "designs": {"initial": design_record("Initial: uniform 6 mm", x_uniform(T_START)),
                    "optimized": design_record("Optimized, nonlinear material", x_opt),
                    "linear": design_record("Optimized assuming a linear material", x_lin)},
        "uniform_feasible": uniform,
        "optimization": {"iterations": int(runs["nonlinear"]["res"].nit), "seconds": runs["nonlinear"]["seconds"],
                         "success": bool(runs["nonlinear"]["res"].success),
                         "saving_vs_uniform_percent": 100 * (1 - m_opt / uniform["mass_kg"]),
                         "stress_margin_percent": 100 * (1 - 1e6 * opt_metrics["stress_MPa"] / cfg["sigma_allow"]),
                         "starts": starts, "kkt": kkt, "mesh_check": mesh_check,
                         "element_peaks": element_peaks, "shape": shape,
                         "linear_analysis_of_optimum": metrics(x_opt, nonlinear=False)[0]},
        "models": models, "nonlinear": nonlinear_summary,
        "project1": p1, "conditioning": cond, "load_path": load_path(x_opt),
        "history": hist, "frames": frames,
        "verification": {"ladder": ladder, **extra, "n_checks": len(ladder),
                         "all_pass": all(r["pass"] for r in ladder)},
        "gradient_cost": cost,
    }
    p1_row = next(r for r in ladder if r["test"].startswith("Project 1"))
    results["verification"]["p1_digits"] = int(-np.log10(max(p1_row["rel_error"], 1e-16)))
    results["conditioning"]["ratio"] = cond["nonlinear"]["kappa"] / cond["linear"]["kappa"]
    results["meta"]["run_minutes"] = (time.perf_counter() - t_begin) / 60

    out = SITE / "assets" / "results.json"
    out.write_text(json.dumps(results, separators=(",", ":"), default=float))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) in {60 * results['meta']['run_minutes']:.0f} s")

    import make_site
    make_site.main()


if __name__ == "__main__":
    main()
