"""Run the whole Project 4 study and regenerate the report site.

    python projects/04_engineering_design/src/run_all.py

Writes site/assets/results.json, then calls make_site.py, which draws the
figures and fills site/index.md from site/index.template.md. Every number on
the site comes from results.json.
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

from plate import SIGMA  # noqa: E402
from problem import CELSIUS, CONFIG, HeatSpreader  # noqa: E402
from verify import run_ladder  # noqa: E402

SITE = Path(__file__).resolve().parents[1] / "site"
PROJECTS = Path(__file__).resolve().parents[2]
T_START = 2.0e-3  # uniform starting thickness, m


def x_uniform(t, cfg=CONFIG):
    n = cfg["tiles_x"] * cfg["tiles_y"]
    return np.full(n, (t - cfg["t_min"]) / (cfg["t_max"] - cfg["t_min"]))


def optimize(model="full", x0=None, cfg=CONFIG, constraints=("chip", "optic")):
    """Minimize mass subject to the temperature limits with SLSQP.

    Scaling: x in [0, 1], mass divided by the mass of the full-thickness plate,
    constraints written as g = KS(theta/theta_limit) - 1 <= 0."""
    hs = HeatSpreader(cfg, model)
    x0 = x_uniform(T_START, cfg) if x0 is None else x0
    m_ref = hs.mass(np.ones(hs.n))
    cache = {}

    def evaluate(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = hs.evaluate(x)
        return cache[key]

    cons = [{"type": "ineq",
             "fun": lambda x, c=c: -evaluate(x)["g_" + c],
             "jac": lambda x, c=c: -evaluate(x)["dg_" + c]} for c in constraints]
    path = [x0.copy()]
    t0 = time.perf_counter()
    res = minimize(lambda x: hs.mass(x) / m_ref, x0, jac=lambda x: hs.mass_gradient() / m_ref,
                   method="SLSQP", bounds=[(0.0, 1.0)] * hs.n, constraints=cons,
                   callback=lambda x: path.append(x.copy()),
                   options={"maxiter": 500, "ftol": 1e-9})
    return {"hs": hs, "x": res.x, "res": res, "path": path, "m_ref": m_ref,
            "seconds": time.perf_counter() - t0, "constraints": constraints}


def metrics(x, cfg=CONFIG, model="full"):
    """Objective and constraints of a design, evaluated in the given model."""
    hs = HeatSpreader(cfg, model)
    e = hs.evaluate(x, gradients=False)
    p, T = hs.plate, e["T"]
    q_conv = p.h * float(np.sum(p.M @ (T - p.T_inf)))
    out = {"mass_g": 1e3 * e["mass"], "heat_convection_W": q_conv,
           "heat_radiation_W": p.heat_out(T) - q_conv,
           "radiation_percent": 100 * (1 - q_conv / p.heat_out(T)), "newton_iterations": len(e["newton"]) - 1,
           "T_mean_C": float(np.sum(p.M @ T)) / (cfg["Lx"] * cfg["Ly"]) - CELSIUS,
           "T_min_C": float(T.min()) - CELSIUS}
    for c in ("chip", "optic"):
        limit = cfg[c]["T_max"]
        out[f"T_{c}_C"] = float(e[f"T_{c}"]) - CELSIUS
        out[f"T_{c}_ks_C"] = float(cfg["T_inf"] + (e[f"g_{c}"] + 1) * (limit - cfg["T_inf"])) - CELSIUS
        out[f"g_{c}"] = float(e[f"g_{c}"])
        out[f"margin_{c}_K"] = float(limit - e[f"T_{c}"])
        out[f"pass_{c}"] = bool(e[f"T_{c}"] <= limit + 1e-6)
    out["feasible"] = out["pass_chip"] and out["pass_optic"]
    return out, hs, T


def design_record(label, x, cfg=CONFIG):
    """Everything the 3D viewer needs for one design (nonlinear model)."""
    m, hs, T = metrics(x, cfg)
    return {"label": label, "metrics": m,
            "t_mm": np.round(1e3 * hs.thickness(x), 4).tolist(),
            "T_C": np.round(T - CELSIUS, 3).tolist()}


def kkt_report(run):
    """Stationarity of the Lagrangian at the optimum, using SLSQP's multipliers."""
    hs, x, m_ref = run["hs"], run["x"], run["m_ref"]
    e = hs.evaluate(x)
    mu = dict(zip(run["constraints"], np.asarray(run["res"].multipliers, float)))
    r = hs.mass_gradient() / m_ref + sum(mu[c] * e["dg_" + c] for c in mu)
    lower, upper = x < 1e-6, x > 1 - 1e-6
    free = ~(lower | upper)
    scale = (hs.mass_gradient() / m_ref)[0]
    out = {"stationarity_free_percent": 100 * float(np.abs(r[free]).max() / scale),
           "n_free": int(free.sum()), "n_lower": int(lower.sum()), "n_upper": int(upper.sum()),
           "stationarity_free": float(np.abs(r[free]).max() / scale) if free.any() else 0.0,
           "min_reduced_gradient_lower": float(r[lower].min() / scale) if lower.any() else 0.0,
           "max_reduced_gradient_upper": float(r[upper].max() / scale) if upper.any() else 0.0,
           "multipliers": {c: float(v) for c, v in mu.items()}}
    # dm*/dT_limit = -mu * m_ref / theta_limit  (grams per kelvin of relaxation)
    for c, v in mu.items():
        theta_limit = hs.cfg[c]["T_max"] - hs.cfg["T_inf"]
        out[f"saving_{c}_g_per_K"] = float(1e3 * v * m_ref / theta_limit)
    return out


def relaxed(cfg, component, delta):
    new = copy.deepcopy(cfg)
    new[component]["T_max"] += delta
    return new


def gradient_cost():
    """Time one adjoint gradient evaluation against forward differences."""
    hs = HeatSpreader()
    x = x_uniform(T_START)
    hs.evaluate(x)  # warm up
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        hs.T_last = None
        e = hs.evaluate(x)
    t_adjoint = (time.perf_counter() - t0) / reps

    step = 1e-6
    t0 = time.perf_counter()
    hs.T_last = None
    base = hs.evaluate(x, gradients=False)
    fd = {c: np.zeros(hs.n) for c in ("chip", "optic")}
    for j in range(hs.n):
        xp = x.copy()
        xp[j] += step
        hs.T_last = None
        ep = hs.evaluate(xp, gradients=False)
        for c in fd:
            fd[c][j] = (ep["g_" + c] - base["g_" + c]) / step
    t_fd = time.perf_counter() - t0
    diff = max(float(np.abs(fd[c] - e["dg_" + c]).max() / np.abs(e["dg_" + c]).max()) for c in fd)
    return {"adjoint_seconds": t_adjoint, "fd_seconds": t_fd, "speedup": t_fd / t_adjoint,
            "n_variables": hs.n, "n_dof": hs.plate.nn, "fd_solves": hs.n + 1,
            "max_difference": diff}


def conditioning(x, refine=None):
    """Condition number of the converged tangent and CG iteration counts, as in Project 2."""
    hs = HeatSpreader(refine=refine)
    p = hs.plate
    T, _ = hs.solve(x)
    KT = p.tangent(T, hs.thickness(x)[hs.tile_of_element])
    A = KT.toarray()
    d = A.diagonal()
    lam = np.linalg.eigvalsh(A)
    lam_jacobi = np.linalg.eigvalsh(A / np.sqrt(np.outer(d, d)))
    out = {"unknowns": p.nn, "element_mm": 1e3 * p.dx, "lambda_min": float(lam[0]),
           "lambda_max": float(lam[-1]), "kappa": float(lam[-1] / lam[0]),
           "kappa_jacobi": float(lam_jacobi[-1] / lam_jacobi[0])}
    for name, M in (("cg_iterations", None), ("pcg_iterations", diags(1 / d))):
        count = [0]
        cg(KT, hs.Q, rtol=1e-6, atol=0.0, M=M, maxiter=20000, callback=lambda _: count.__setitem__(0, count[0] + 1))
        out[name] = count[0]
    # every node loses heat to ambient, which bounds the smallest eigenvalue from below
    h_r = p.eps * SIGMA * (T**2 + p.T_inf**2) * (T + p.T_inf)
    out["lambda_min_estimate"] = float((p.h + h_r.min()) * p.area)
    return out


def earlier_projects(x_opt):
    """Quantities that tie this project to Projects 1 to 3."""
    hs = HeatSpreader()
    e = hs.evaluate(x_opt)
    signs = {c: {"lower": int(np.sum(e["dg_" + c] < 0)), "raise": int(np.sum(e["dg_" + c] > 0))}
             for c in ("chip", "optic")}
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        hs.T_last = None
        hs.solve(x_opt)
    solve_ms = 1e3 * (time.perf_counter() - t0) / reps

    with open(PROJECTS / "02_gradient_descent" / "results" / "conditioning.csv") as f:
        p2 = {row["case"]: row for row in csv.DictReader(f)}["optimized_120x40_Emin_1e-09"]
    with open(PROJECTS / "02_gradient_descent" / "results" / "solvers.csv") as f:
        p2_solvers = {(row["case"], row["method"]): row for row in csv.DictReader(f)}
    p2_cg = p2_solvers[("optimized_120x40_Emin_1e-09", "CG")]
    p2_pcg = p2_solvers[("optimized_120x40_Emin_1e-09", "Jacobi-PCG")]
    p3 = json.loads((PROJECTS / "03_neural_operator" / "results" / "summary.json").read_text())["cost"]
    x_start = x_uniform(T_START)
    uniform, coarse = conditioning(x_start), conditioning(x_start, refine=1)
    optimized = conditioning(x_opt)
    return {
        "gradient_signs": signs, "solve_ms": solve_ms,
        "conditioning": {"uniform": uniform, "uniform_coarse": coarse, "optimized": optimized,
                         "mesh_growth": uniform["kappa"] / coarse["kappa"],
                         "contrast_growth": optimized["kappa"] / uniform["kappa"],
                         "thickness_contrast": CONFIG["t_max"] / CONFIG["t_min"]},
        "project2": {"unknowns": int(p2["free_dofs"]), "kappa": float(p2["kappa"]),
                     "kappa_jacobi": float(p2["jacobi_kappa"]), "emin": float(p2["emin"]),
                     "cg_iterations": int(p2_cg["iterations"]), "cg_converged": p2_cg["converged"] == "True",
                     "pcg_iterations": int(p2_pcg["iterations"])},
        "project3": {"fe_solve_seconds": p3["fe_solve_seconds_median"],
                     "cost_ratio": p3["per_design_cost_ratio_lbfgs"]},
    }


def main():
    t_begin = time.perf_counter()
    cfg = CONFIG
    Tinf, area = cfg["T_inf"], cfg["Lx"] * cfg["Ly"]
    power = cfg["chip"]["power"] + cfg["optic"]["power"]

    print("verification ladder ...")
    ladder, extra = run_ladder()
    for r in ladder:
        print(f"  {r['id']} {'ok  ' if r['pass'] else 'FAIL'} {r['test']}: {r['quantity']}  "
              f"{r['computed']:.8g} vs {r['exact']:.8g}  ({r['rel_error']:.1e})")

    print("optimization, nonlinear model ...")
    runs = {"full": optimize("full")}
    x_opt = runs["full"]["x"]
    rng = np.random.default_rng(0)
    start_points = {"uniform 2 mm": None,
                    "uniform 1 mm (lower bound)": x_uniform(cfg["t_min"]),
                    "uniform 8 mm (upper bound)": x_uniform(cfg["t_max"]),
                    "random": rng.uniform(0, 1, x_opt.size)}
    starts = []
    for name, x0 in start_points.items():
        run = runs["full"] if x0 is None else optimize("full", x0)
        m, _, _ = metrics(run["x"])
        starts.append({"start": name, "iterations": int(run["res"].nit), "seconds": run["seconds"],
                       "mass_g": m["mass_g"], "T_chip_C": m["T_chip_C"], "T_optic_C": m["T_optic_C"],
                       "max_thickness_difference_mm": float(
                           1e3 * (cfg["t_max"] - cfg["t_min"]) * np.abs(run["x"] - x_opt).max())})
        print(f"  start {name}: {m['mass_g']:.2f} g in {run['res'].nit} iterations")

    print("optimization, linearized radiation ...")
    runs["linearized"] = optimize("linearized")
    x_lin = runs["linearized"]["x"]

    print("multiplier checks ...")
    kkt = kkt_report(runs["full"])
    opt_metrics = metrics(x_opt)[0]
    m_opt = opt_metrics["mass_g"]
    kkt["relaxation_K"] = 0.2
    for c in ("chip", "optic"):  # relax one limit slightly, re-optimize, compare with the multiplier
        cfg_relaxed = relaxed(cfg, c, kkt["relaxation_K"])
        m_relaxed = metrics(optimize("full", cfg=cfg_relaxed)["x"], cfg_relaxed)[0]["mass_g"]
        kkt[f"reoptimized_saving_{c}_g_per_K"] = (m_opt - m_relaxed) / kkt["relaxation_K"]
    no_optic = optimize("full", constraints=("chip",))
    m_no_optic, _, _ = metrics(no_optic["x"])
    kkt["mass_without_optic_limit_g"] = m_no_optic["mass_g"]
    kkt["T_optic_without_limit_C"] = m_no_optic["T_optic_C"]
    kkt["cost_of_optic_limit_g"] = m_opt - m_no_optic["mass_g"]
    kkt["saving_chip_percent_per_K"] = 100 * kkt["saving_chip_g_per_K"] / m_opt

    # thinnest uniform plate that satisfies the module constraint, for reference
    lo, hi = cfg["t_min"], cfg["t_max"]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if metrics(x_uniform(mid))[0]["g_chip"] <= 0 else (mid, hi)
    uniform_chip = metrics(x_uniform(hi))[0]
    uniform_chip["thickness_mm"] = 1e3 * hi

    # models compared
    h_r_lin = 4 * cfg["eps"] * SIGMA * Tinf**3
    models = {}
    for name, x in (("full", x_opt), ("linearized", x_lin)):
        own, _, _ = metrics(x, model=name)
        true, _, _ = metrics(x, model="full")
        models[name] = {"mass_g": own["mass_g"], "iterations": int(runs[name]["res"].nit),
                        "seconds": runs[name]["seconds"], "success": bool(runs[name]["res"].success),
                        "own_model": own, "nonlinear_model": true}
    thick_none, _, _ = metrics(x_uniform(cfg["t_max"]), model="none")
    T_chip_peak = opt_metrics["T_chip_C"] + CELSIUS
    T_cold = opt_metrics["T_min_C"] + CELSIUS
    h_r = lambda T: cfg["eps"] * SIGMA * (T**2 + Tinf**2) * (T + Tinf)
    nonlinear = {
        "h_r_linearized": h_r_lin, "h_r_at_chip": h_r(T_chip_peak), "h_r_at_coldest": h_r(T_cold),
        "mean_T_linearized_C": Tinf + power / ((cfg["h"] + h_r_lin) * area) - CELSIUS,
        "mean_T_convection_only_C": Tinf + power / (cfg["h"] * area) - CELSIUS,
        "convection_only_thickest_plate": thick_none,
        "extra_mass_linearized_g": models["linearized"]["mass_g"] - models["full"]["mass_g"],
        "extra_mass_linearized_percent": 100 * (models["linearized"]["mass_g"] / models["full"]["mass_g"] - 1),
    }

    # mesh check of the optimized design
    mesh_check = []
    for rf in (1, 2, 4, 8):
        hs = HeatSpreader(refine=rf)
        e = hs.evaluate(x_opt, gradients=False)
        mesh_check.append({"element_mm": 1e3 * hs.plate.dx, "elements": hs.plate.ne,
                           "T_chip_C": float(e["T_chip"]) - CELSIUS,
                           "T_optic_C": float(e["T_optic"]) - CELSIUS})

    print("gradient cost ...")
    cost = gradient_cost()

    # optimization history of the nonlinear run
    path = runs["full"]["path"]
    hist = {"iteration": [], "mass_g": [], "T_chip_C": [], "T_optic_C": []}
    for i, x in enumerate(path):
        m, _, _ = metrics(x)
        hist["iteration"].append(i)
        for k in ("mass_g", "T_chip_C", "T_optic_C"):
            hist[k].append(round(m[k], 4))
    keep = sorted(set(np.unique(np.round(np.linspace(0, np.sqrt(len(path) - 1), 36) ** 2).astype(int))))
    frames = []
    for i in keep:
        rec = design_record(f"iteration {i}", path[i])
        frames.append({"iteration": int(i), "t_mm": rec["t_mm"],
                       "T_C": np.round(rec["T_C"], 2).tolist(),
                       "mass_g": rec["metrics"]["mass_g"],
                       "T_chip_C": rec["metrics"]["T_chip_C"], "T_optic_C": rec["metrics"]["T_optic_C"]})

    hs = runs["full"]["hs"]
    box = lambda c: {"x0": c["center"][0] - c["side"] / 2, "x1": c["center"][0] + c["side"] / 2,
                     "y0": c["center"][1] - c["side"] / 2, "y1": c["center"][1] + c["side"] / 2,
                     "power_W": c["power"], "limit_C": c["T_max"] - CELSIUS}
    results = {
        "meta": {"command": "python projects/04_engineering_design/src/run_all.py",
                 "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                 "optimizer": "SLSQP (scipy.optimize.minimize)"},
        "config": {"Lx_mm": 1e3 * cfg["Lx"], "Ly_mm": 1e3 * cfg["Ly"], "tiles_x": cfg["tiles_x"],
                   "tiles_y": cfg["tiles_y"], "tile_mm": 1e3 * cfg["Lx"] / cfg["tiles_x"],
                   "refine": cfg["refine"], "nx": hs.plate.nx, "ny": hs.plate.ny,
                   "element_mm": 1e3 * hs.plate.dx, "n_nodes": hs.plate.nn, "n_elements": hs.plate.ne,
                   "n_variables": hs.n, "t_min_mm": 1e3 * cfg["t_min"], "t_max_mm": 1e3 * cfg["t_max"],
                   "t_start_mm": 1e3 * T_START, "k": cfg["k"], "rho": cfg["rho"], "eps": cfg["eps"],
                   "h": cfg["h"], "T_inf_C": Tinf - CELSIUS, "power_W": power, "ks_rho": cfg["ks_rho"],
                   "mass_full_plate_g": 1e3 * runs["full"]["m_ref"],
                   "biot": (cfg["h"] + h_r(T_chip_peak)) * cfg["t_max"] / cfg["k"],
                   "ks_bound_K": {c: float(np.log(hs.nodes[c].size) / cfg["ks_rho"]
                                           * (cfg[c]["T_max"] - Tinf)) for c in ("chip", "optic")},
                   "ks_nodes": {c: int(hs.nodes[c].size) for c in ("chip", "optic")}},
        "components": {"chip": box(cfg["chip"]), "optic": box(cfg["optic"])},
        "designs": {"initial": design_record("Initial: uniform 2 mm", x_uniform(T_START)),
                    "optimized": design_record("Optimized", x_opt),
                    "linearized": design_record("Optimized with linearized radiation", x_lin)},
        "uniform_chip_feasible": uniform_chip,
        "optimization": {"iterations": int(runs["full"]["res"].nit), "seconds": runs["full"]["seconds"],
                         "message": str(runs["full"]["res"].message),
                         "success": bool(runs["full"]["res"].success),
                         "saving_vs_initial_percent": 100 * (1 - m_opt / metrics(x_uniform(T_START))[0]["mass_g"]),
                         "saving_vs_uniform_chip_percent": 100 * (1 - m_opt / uniform_chip["mass_g"]),
                         "starts": starts, "kkt": kkt, "mesh_check": mesh_check},
        "models": models, "nonlinear": nonlinear,
        "history": hist, "frames": frames,
        "verification": {"ladder": ladder, **extra},
        "gradient_cost": cost,
        "earlier": earlier_projects(x_opt),
    }
    results["verification"]["all_pass"] = all(r["pass"] for r in ladder)
    results["verification"]["n_checks"] = len(ladder)
    results["meta"]["run_minutes"] = (time.perf_counter() - t_begin) / 60

    out = SITE / "assets" / "results.json"
    out.write_text(json.dumps(results, separators=(",", ":"), default=float))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) in {60 * results['meta']['run_minutes']:.0f} s")

    import make_site
    make_site.main()


if __name__ == "__main__":
    main()
