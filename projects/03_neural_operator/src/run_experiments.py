"""Project 3 experiments: energy-trained PINN for the half-MBB equilibrium solve.

Runs every training configuration used in the report, two at a time on CPU,
and writes curated results to ../results:

  results/runs.csv            one row per run (final metrics, timings, config)
  results/histories/<id>.csv  logged loss components and errors during training
  results/fields.npz          final displacement vectors used in the figures
  results/reference.json      FE reference solves and timings (P1 baseline)
  results/metadata.json       versions, seeds, hardware

Run from the repository root:  python projects/03_neural_operator/src/run_experiments.py
Wall time is about one hour on a 2-core laptop CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent

OC_ITERS = (0, 2, 5, 10, 20, 90)        # design-maturity knob (P1 OC update count)
EMINS = (1e-1, 1e-2, 1e-3, 1e-6, 1e-9)   # stiffness-contrast knob (P2 family A)
MESHES = (8, 16, 40)                     # mesh knob on the uniform layout (P2 family B)
BC_WEIGHTS = (1e-1, 1e1, 1e3)            # soft Dirichlet penalty springs
LOAD_WIDTH = 10.0                        # distributed-load control for the mesh knob
MESH_DIST = (8, 16, 40, 80)              # mesh sweep with the distributed load
ADAM = dict(steps=20000, lr=1e-3, lr_final=1e-4, log_every=500)
LBFGS_ITERS = 500
SAVE_FIELDS = {"oc90_adam_s0", "oc90_lbfgs_s0", "oc90_sup_s0", "oc0_adam_s0", "oc0_sup_s0",
               "oc0_lbfgs_s0"}


def configs():
    runs = []
    for k in OC_ITERS:
        for method in ("adam", "lbfgs", "sup"):
            runs.append(dict(id=f"oc{k}_{method}_s0", group="oc", oc=k, emin=1e-9,
                             ny=40, method=method, seed=0, bc="hard"))
    for emin in EMINS[:-1]:
        for method in ("adam", "lbfgs"):
            runs.append(dict(id=f"emin{emin:.0e}_{method}_s0", group="emin", oc=90, emin=emin,
                             ny=40, method=method, seed=0, bc="hard"))
    for ny in MESHES[:-1]:
        runs.append(dict(id=f"mesh{ny}_adam_s0", group="mesh", oc=0, emin=1e-9, ny=ny,
                         method="adam", seed=0, bc="hard"))
    for ny in MESH_DIST:   # same mesh sweep with a distributed load (no point singularity)
        runs.append(dict(id=f"meshdist{ny}_adam_s0", group="meshdist", oc=0, emin=1e-9, ny=ny,
                         method="adam", seed=0, bc="hard", load_width=LOAD_WIDTH))
    for w in BC_WEIGHTS:
        runs.append(dict(id=f"bc{w:.0e}_adam_s0", group="bc", oc=90, emin=1e-9, ny=40,
                         method="adam", seed=0, bc=w))
    for seed in (1, 2):
        for method in ("adam", "lbfgs"):
            runs.append(dict(id=f"oc90_{method}_s{seed}", group="seed", oc=90, emin=1e-9,
                             ny=40, method=method, seed=seed, bc="hard"))
    runs.append(dict(id="oc90_adam40k_s0", group="budget", oc=90, emin=1e-9, ny=40,
                     method="adam40k", seed=0, bc="hard"))
    for k in (0, 90):   # output scale from the design's mean modulus instead of E0
        runs.append(dict(id=f"oc{k}_adam_meanE_s0", group="scale", oc=k, emin=1e-9, ny=40,
                         method="adam", seed=0, bc="hard", scale="mean_modulus"))
    cost = {"adam40k": 4, "lbfgs": 3, "adam": 2, "sup": 2}
    return sorted(runs, key=lambda r: -cost[r["method"]] * (r["ny"] / 40) ** 2)


_SNAPSHOTS = None


def density_for(cfg):
    import pinn_core as pc
    global _SNAPSHOTS
    if cfg["ny"] != 40:
        return None                                    # uniform 0.5 on coarser meshes
    if _SNAPSHOTS is None:
        _SNAPSHOTS = pc.oc_snapshots(OC_ITERS)
    return _SNAPSHOTS[cfg["oc"]]["rho"]


def run(cfg):
    import torch
    torch.set_num_threads(1)
    import pinn_core as pc
    pc.set_precision(torch.float32)
    ny = cfg["ny"]
    prob = pc.build_problem(3 * ny, ny, rho=density_for(cfg), emin=cfg["emin"], solve_repeats=1,
                            load_width=cfg.get("load_width"))
    hard = cfg["bc"] == "hard"
    uc = pc.beam_scale(prob)
    if cfg.get("scale") == "mean_modulus":
        uc /= float(prob.modulus.mean())     # Euler-Bernoulli with the design's mean modulus
    model = pc.Model(prob, pc.FieldNet(seed=cfg["seed"]), hard_bc=hard, out_scale=uc)
    bc_weight = 0.0 if hard else float(cfg["bc"])
    method = cfg["method"]
    if method in ("adam", "adam40k"):
        kw = dict(ADAM, steps=2 * ADAM["steps"]) if method == "adam40k" else ADAM
        hist, u, cost = pc.train(prob, model, bc_weight=bc_weight, **kw)
    elif method == "sup":
        hist, u, cost = pc.train(prob, model, mode="supervised", **ADAM)
    elif method == "lbfgs":
        h1, _, c1 = pc.train(prob, model, **dict(ADAM, steps=ADAM["steps"] // 2))
        pc.set_precision(torch.float64)            # line searches need float64
        model.to(torch.float64)
        h2, u, c2 = pc.train(prob, model, optimizer="lbfgs", lbfgs_iters=LBFGS_ITERS, log_every=25)
        for row in h2:
            row["iter"] += ADAM["steps"] // 2
            row["seconds"] += c1["train_seconds"]
        hist = h1 + h2[1:]
        cost = {"train_seconds": c1["train_seconds"] + c2["train_seconds"],
                "inference_seconds": c1["inference_seconds"]}  # float32 inference
    else:
        raise ValueError(method)
    final = dict(pc.metrics(prob, u))
    final.update(energy_gap=hist[-1]["energy_gap"], **cost,
                 n_params=sum(p.numel() for p in model.parameters()),
                 compliance_ref=prob.compliance_ref)
    return cfg, hist, final, (u if cfg["id"] in SAVE_FIELDS else None)


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def reference_block(out):
    """P1 baseline: FE direct solve of every case, with median timings."""
    import pinn_core as pc
    global _SNAPSHOTS
    snaps = _SNAPSHOTS = pc.oc_snapshots(OC_ITERS)   # inherited by forked workers
    p1_final = pc.load_p1_density()
    assert np.array_equal(snaps[90]["rho"], p1_final), "OC re-run must reproduce P1"
    rows = []
    for k in OC_ITERS:
        p = pc.build_problem(rho=snaps[k]["rho"], emin=1e-9, solve_repeats=7)
        resid = np.linalg.norm(p.K_free @ p.u_ref[p.free] - p.force[p.free])
        rows.append({"case": f"oc{k}", "oc": k, "emin": 1e-9, "ny": 40,
                     "compliance": p.compliance_ref, "gray_fraction": snaps[k]["gray_fraction"],
                     "solid_fraction": float(np.mean(snaps[k]["rho"] >= 0.5)),
                     "relative_residual": float(resid / np.linalg.norm(p.force[p.free])),
                     "solve_seconds_median": p.solve_seconds})
    for emin in EMINS[:-1]:
        p = pc.build_problem(rho=p1_final, emin=emin, solve_repeats=7)
        rows.append({"case": f"emin{emin:.0e}", "oc": 90, "emin": emin, "ny": 40,
                     "compliance": p.compliance_ref, "solve_seconds_median": p.solve_seconds})
    for ny in MESHES[:-1]:
        p = pc.build_problem(3 * ny, ny, emin=1e-9, solve_repeats=7)
        rows.append({"case": f"mesh{ny}", "oc": 0, "emin": 1e-9, "ny": ny,
                     "compliance": p.compliance_ref, "solve_seconds_median": p.solve_seconds})
    for ny in MESH_DIST:
        p = pc.build_problem(3 * ny, ny, emin=1e-9, solve_repeats=7, load_width=LOAD_WIDTH)
        rows.append({"case": f"meshdist{ny}", "oc": 0, "emin": 1e-9, "ny": ny,
                     "load_width": LOAD_WIDTH, "compliance": p.compliance_ref,
                     "solve_seconds_median": p.solve_seconds})
    json.dump({"p1_final_compliance_check": rows[len(OC_ITERS) - 1]["compliance"], "cases": rows},
              (out / "reference.json").open("w"), indent=2)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=PROJECT / "results")
    ap.add_argument("--only", nargs="*", help="run ids to run (default: all)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, for testing the pipeline")
    ap.add_argument("--reference-only", action="store_true",
                    help="only redo the FE reference solves and timings (on an idle machine)")
    args = ap.parse_args()
    if args.smoke:
        global ADAM, LBFGS_ITERS
        ADAM = dict(ADAM, steps=40, log_every=10)
        LBFGS_ITERS = 3
    out = args.output.resolve()
    (out / "histories").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    reference_block(out)
    if args.reference_only:
        return
    todo = [c for c in configs() if not args.only or c["id"] in args.only]
    rows, fields = [], {}
    if args.only and (out / "runs.csv").exists():      # update an existing run table
        with (out / "runs.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if r["id"] not in args.only]
        if (out / "fields.npz").exists():
            fields = dict(np.load(out / "fields.npz"))
    from multiprocessing import get_context
    with get_context("fork").Pool(args.workers) as pool:
        for cfg, hist, final, u in pool.imap_unordered(run, todo):
            csv_write(out / "histories" / f"{cfg['id']}.csv", hist)
            rows.append({**cfg, **final})
            if u is not None:
                fields[cfg["id"]] = u
            print(f"[{time.perf_counter() - t0:7.0f}s] {cfg['id']:<22} struct {final['rel_err_struct']:.3e}"
                  f"  comp {final['rel_compliance_err']:.3e}  train {final['train_seconds']:.0f}s",
                  flush=True)
            rows.sort(key=lambda r: r["id"])
            csv_write(out / "runs.csv", rows)
    if fields:
        np.savez_compressed(out / "fields.npz", **fields)
    import torch, scipy, matplotlib
    meta = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__, "platform": platform.platform(),
            "cpu_threads_per_run": 1, "parallel_runs": args.workers,
            "processor": platform.processor() or "unknown",
            "network": "MLP 2-64-64-64-64-2, tanh, Xavier-normal init, hard BCs",
            "adam": ADAM, "lbfgs": "10k Adam steps (float32), then 500 L-BFGS iterations "
                                   "(max_iter 20, history 50, strong Wolfe, float64)",
            "seeds": "network seed in run id (s0, s1, s2)",
            "density_sha256": hashlib.sha256(
                (PROJECT.parent / "01_problem_formulation/results/final_density.csv").read_bytes()
            ).hexdigest(),
            "wall_seconds": time.perf_counter() - t0}
    if not args.only:
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
