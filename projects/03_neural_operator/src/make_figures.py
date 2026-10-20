"""Figures and summary tables for the Project 3 report.

Reads ../results (written by run_experiments.py) and writes ../figures plus
results/summary.json. Run from the repository root:

    python projects/03_neural_operator/src/make_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent
P2_RESULTS = PROJECT.parent / "02_gradient_descent" / "results"

import pinn_core as pc  # noqa: E402
from run_experiments import ADAM, EMINS, MESH_DIST, MESHES, OC_ITERS  # noqa: E402

ADAM_HALF = ADAM["steps"] // 2
LOW_M = 3          # wavelength >= 40 = beam height

C_ADAM, C_LBFGS, C_SUP, C_REF = "#ba5a38", "#167d8d", "#7a5aa6", "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.18})


def read_csv(path):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def history(res, run_id):
    rows = read_csv(res / "histories" / f"{run_id}.csv")
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def kappa(prob):
    A = prob.K_free
    v0 = np.random.default_rng(598).normal(size=A.shape[0])
    lo = eigsh(A, k=1, sigma=0.0, which="LM", v0=v0, tol=1e-10, return_eigenvectors=False)[0]
    hi = eigsh(A, k=1, which="LA", v0=v0, tol=1e-10, return_eigenvectors=False)[0]
    return float(hi / lo)


def node_grid(prob, u):
    ny, nx = prob.ny, prob.nx
    return (u[0::2].reshape((ny + 1, nx + 1), order="F"),
            u[1::2].reshape((ny + 1, nx + 1), order="F"))


def structure_nodes(prob):
    mask = np.zeros(prob.ndof // 2, bool)
    mask[np.unique(prob.edof[prob.rho >= 0.5][:, 0::2] // 2)] = True
    return mask.reshape((prob.ny + 1, prob.nx + 1), order="F")


def x_spectrum(fields):
    """Power spectrum along the beam axis: FFT of every node row in x, summed
    over rows and both displacement components. Mode m has m cycles across the
    beam length (m = 60 is the grid scale on the 120-element mesh)."""
    power = 0.0
    for f in fields:
        F = np.fft.rfft(f, axis=1)
        power = power + (np.abs(F) ** 2).sum(axis=0)
    power[1:-1] *= 2.0                                 # one-sided spectrum
    return np.arange(power.size), power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    ap.add_argument("--figures", type=Path, default=PROJECT / "figures")
    args = ap.parse_args()
    res, figdir = args.results.resolve(), args.figures.resolve()
    figdir.mkdir(exist_ok=True)
    runs = {r["id"]: r for r in read_csv(res / "runs.csv")}
    ref = json.loads((res / "reference.json").read_text())["cases"]
    ref_by = {r["case"]: r for r in ref}
    fields = np.load(res / "fields.npz")
    snaps = pc.oc_snapshots(OC_ITERS)
    summary = {}

    # ---------------- Figure 1: the knob (design snapshots) ----------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 3.9), layout="constrained")
    for ax, k in zip(axes.ravel(), OC_ITERS):
        ax.imshow(snaps[k]["rho"].reshape((40, 120), order="F"), cmap="gray_r", vmin=0, vmax=1,
                  extent=(0, 120, 0, 40))
        ax.set_title(f"k = {k}:  C = {ref_by[f'oc{k}']['compliance']:.0f},  "
                     f"gray {100 * snaps[k]['gray_fraction']:.0f}%", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Difficulty knob: the Project 1 design after k OC updates (equilibrium solved on each)",
                 fontsize=11)
    fig.savefig(figdir / "knob_designs.png", dpi=170)
    plt.close(fig)

    # ---------------- Figure 2: loss components (P2) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    C = ref_by["oc90"]["compliance"]
    for ax, rid, title in ((axes[0], "oc90_adam_s0", "(a) Energy loss, Adam"),
                           (axes[1], "oc90_lbfgs_s0", "(b) Energy loss, Adam then L-BFGS")):
        h = history(res, rid)
        it = h["seconds"]
        ax.loglog(it[1:], h["energy_gap"][1:], color="k", lw=2, label=r"loss gap $(\Pi-\Pi^\star)/|\Pi^\star|$")
        ax.loglog(it[1:], np.abs(h["strain_energy"][1:] - C / 2) / (C / 2), color="#2a6fbb",
                  label=r"strain energy $|U-U^\star|/U^\star$")
        ax.loglog(it[1:], np.abs(h["load_work"][1:] - C) / C, color="#d08a1e", ls="--",
                  label=r"load work $|W-W^\star|/W^\star$")
        ax.loglog(it[1:], h["rel_err_struct"][1:], color=C_ADAM if "adam" in rid else C_LBFGS, lw=2,
                  label="displacement error (structure)")
        if "lbfgs" in rid:
            t_switch = it[np.argmax(h["iter"] > ADAM_HALF)] if np.any(h["iter"] > ADAM_HALF) else it[-1]
            ax.axvline(t_switch, color="0.5", ls=":", lw=1)
            ax.text(t_switch * 1.08, 3, "switch to\nL-BFGS", fontsize=8, color="0.35")
        ax.set(xlabel="Wall-clock training time (s)", ylim=(1e-4, 20), title=title)
    axes[0].legend(fontsize=7.6, loc="lower left")
    ax = axes[2]
    for w, color in zip((1e-1, 1e1, 1e3), ("#9cb6d6", "#5b86bf", "#1d3f75")):
        h = history(res, f"bc{w:.0e}_adam_s0")
        ax.loglog(h["iter"][1:], h["rel_err_struct"][1:], color=color, label=fr"soft BC, $\beta={w:g}$")
    h = history(res, "oc90_adam_s0")
    ax.loglog(h["iter"][1:], h["rel_err_struct"][1:], color=C_ADAM, lw=2, label="hard BC")
    ax.set(xlabel="Adam step", ylabel="Displacement error (structure)", ylim=(1e-2, 3),
           title="(c) Dirichlet penalty vs exact BCs")
    ax.legend(fontsize=8)
    fig.savefig(figdir / "loss_components.png", dpi=170)
    plt.close(fig)

    # ---------------- Figure 3: error vs knobs (P3) ----------------
    kap_rows = []
    p2 = {r["case"]: r for r in read_csv(P2_RESULTS / "conditioning.csv")}
    final_rho = pc.load_p1_density()
    for emin in EMINS:
        prob = pc.build_problem(rho=final_rho, emin=emin, solve_repeats=1)
        kap_rows.append({"emin": emin, "kappa": kappa(prob)})
    for e, key in ((1e-3, "optimized_120x40_Emin_1e-03"), (1e-6, "optimized_120x40_Emin_1e-06"),
                   (1e-9, "optimized_120x40_Emin_1e-09")):
        mine = next(r["kappa"] for r in kap_rows if r["emin"] == e)
        assert abs(mine / p2[key]["kappa"] - 1) < 1e-3, "kappa must match Project 2"
    with (res / "conditioning_emin.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["emin", "kappa"])
        w.writeheader(); w.writerows(kap_rows)

    def rid(group_key, method, seed=0):
        return f"{group_key}_{method}_s{seed}"

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), layout="constrained")
    ax = axes[0]
    x = np.arange(len(OC_ITERS))
    for method, color, label in (("adam", C_ADAM, "energy loss, Adam"),
                                 ("lbfgs", C_LBFGS, "energy loss, Adam then L-BFGS"),
                                 ("sup", C_SUP, "supervised fit to FE (same net, Adam)")):
        y = [runs[rid(f"oc{k}", method)]["rel_err_struct"] for k in OC_ITERS]
        ax.semilogy(x, y, "o-", color=color, label=label)
    if "oc0_adam_meanE_s0" in runs:
        xs = [x[0], x[-1]]
        ax.semilogy(xs, [runs["oc0_adam_meanE_s0"]["rel_err_struct"], runs["oc90_adam_meanE_s0"]["rel_err_struct"]],
                    "D", mfc="none", color=C_ADAM, ms=7, ls="none",
                    label="Adam, output scale from mean modulus")
    ax.set_xticks(x, [f"{k}\n{100 * snaps[k]['gray_fraction']:.0f}%" for k in OC_ITERS])
    ax.set(xlabel="OC update k  /  gray fraction", ylabel="Relative displacement error (structure)",
           title="(a) Design-maturity knob", ylim=(1e-5, 4))
    ax.legend(fontsize=7.5, loc="upper center")

    ax = axes[1]
    em = np.array(EMINS)
    oc_key = lambda e: "oc90" if e == 1e-9 else f"emin{e:.0e}"
    for method, color, label in (("adam", C_ADAM, "Adam"), ("lbfgs", C_LBFGS, "Adam then L-BFGS")):
        ax.loglog(em, [runs[rid(oc_key(e), method)]["rel_err_struct"] for e in em], "o-",
                  color=color, label=f"PINN, {label}")
    ax.set(xlabel=r"Void modulus $E_{\min}$ (contrast grows to the right)",
           ylabel="Relative displacement error (structure)", ylim=(1e-4, 0.5),
           title=r"(b) Stiffness-contrast knob (Project 2, family A)")
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.loglog(em, [r["kappa"] for r in kap_rows], "s--", color=C_REF, label=r"$\kappa(K)$ of the FE matrix")
    ax2.set_ylabel(r"$\kappa(K)$", color=C_REF); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    ax = axes[2]
    nys = np.array(MESHES)
    mesh_key = lambda ny: "oc0" if ny == 40 else f"mesh{ny}"
    ax.loglog(nys, [runs[rid(mesh_key(n), "adam")]["rel_err_struct"] for n in nys], "o-",
              color=C_ADAM, label="PINN, Adam, point load")
    nd = np.array([n for n in MESH_DIST if f"meshdist{n}_adam_s0" in runs])
    ax.loglog(nd, [runs[rid(f"meshdist{n}", "adam")]["rel_err_struct"] for n in nd], "o--",
              color=C_ADAM, mfc="white", label="PINN, Adam, load spread over 10 units")
    ax.set(xlabel=r"Mesh resolution $n_y$ (uniform layout)", ylabel="Relative displacement error",
           ylim=(1e-4, 0.5), title="(c) Mesh knob (Project 2, family B)")
    ax2 = ax.twinx()
    kap_mesh = {}
    for n in MESH_DIST:
        kap_mesh[n] = kappa(pc.build_problem(3 * n, n, emin=1e-9, solve_repeats=1))
        if f"uniform_{3 * n}x{n}" in p2:
            assert abs(kap_mesh[n] / p2[f"uniform_{3 * n}x{n}"]["kappa"] - 1) < 1e-3
    ax2.loglog(list(kap_mesh), list(kap_mesh.values()), "s--", color=C_REF, label=r"$\kappa(K)$ of the FE matrix")
    ax.set_xticks(MESH_DIST, [str(n) for n in MESH_DIST])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_ylabel(r"$\kappa(K)$", color=C_REF); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.savefig(figdir / "error_vs_knobs.png", dpi=170)
    plt.close(fig)
    summary["kappa_emin"] = kap_rows
    summary["kappa_mesh"] = {int(k): v for k, v in kap_mesh.items()}

    # ---------------- Figure 4: where is the error (P4) ----------------
    prob = pc.build_problem(rho=final_rho, emin=1e-9, solve_repeats=1)
    smask = structure_nodes(prob)
    urx, ury = node_grid(prob, prob.u_ref)
    umax = np.sqrt(urx ** 2 + ury ** 2)[smask].max()
    cases = (("oc90_adam_s0", "Energy loss, Adam", C_ADAM),
             ("oc90_sup_s0", "Supervised fit to FE", C_SUP),
             ("oc90_lbfgs_s0", "Energy loss, Adam then L-BFGS", C_LBFGS))
    fig = plt.figure(figsize=(12, 6.4), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    spec_rows = []
    from matplotlib.colors import LogNorm
    for i, (key, label, color) in enumerate(cases):
        ex, ey = node_grid(prob, fields[key] - prob.u_ref)
        mag = np.ma.array(np.sqrt(ex ** 2 + ey ** 2) / umax, mask=~smask)
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(np.maximum(mag, 1e-5), extent=(0, 120, 0, 40), origin="upper", cmap="magma",
                       norm=LogNorm(1e-4, 3e-1))
        ax.set_title(f"({'abc'[i]}) {label}\ndisplacement {runs[key]['rel_err_struct']:.2%}, "
                     f"compliance {runs[key]['rel_compliance_err']:.2%}", fontsize=9.5)
        ax.set_xticks([0, 60, 120]); ax.set_yticks([0, 40]); ax.grid(False)
    fig.colorbar(im, ax=fig.axes[:3], shrink=0.75, label=r"$|u-u^\star|$ / max $|u^\star|$")
    # spectra: optimized (structure-masked) and uniform layouts
    for j, (layout, title) in enumerate((("oc90", "(d) Error spectrum, optimized design (structure nodes)"),
                                         ("oc0", "(e) Error spectrum, uniform design (k = 0)"))):
        ax = fig.add_subplot(gs[1, 0:2] if j == 0 else gs[1, 2])
        p = prob if layout == "oc90" else pc.build_problem(rho=snaps[0]["rho"], emin=1e-9, solve_repeats=1)
        m = smask if layout == "oc90" else np.ones_like(smask)
        rx, ry = node_grid(p, p.u_ref)
        mm, pref = x_spectrum((rx * m, ry * m))
        norm = pref.sum()
        ax.semilogy(mm, pref / norm, color="k", lw=1.1, ls="--", zorder=5, label="reference solution itself")
        for key0, label, color in cases:
            key = key0.replace("oc90", layout)
            if key not in fields:
                continue
            ex, ey = node_grid(p, fields[key] - p.u_ref)
            mm, pe = x_spectrum((ex * m, ey * m))
            low = float(pe[mm <= LOW_M].sum() / pe.sum())
            spec_rows.append({"case": key, "low_mode_share": low,
                              "error_power_over_reference": float(pe.sum() / norm)})
            ax.semilogy(mm, pe / norm, color=color, lw=1.8, marker="o", ms=2.5,
                        label=f"{label}: {100 * low:.0f}%")
        ax.axvspan(-0.5, LOW_M + 0.5, color="0.9", zorder=0)
        ax.set(xlabel="Mode number m along the beam (60 = grid scale)",
               ylabel="Power / reference total power", title=title, ylim=(1e-12, 3), xlim=(-1, 61))
        ax.legend(fontsize=7.4, loc="lower left" if j == 0 else "upper right",
                  title=f"error power share at m ≤ {LOW_M} (shaded)", title_fontsize=7.4)
    fig.savefig(figdir / "where_is_the_error.png", dpi=170)
    plt.close(fig)
    summary["spectrum"] = spec_rows

    # error in the eigenbasis of K, uniform design (no void, so K's softest modes are
    # deformations of the whole beam; cf. Project 2, softest mode = 93% of compliance)
    p0 = pc.build_problem(rho=snaps[0]["rho"], emin=1e-9, solve_repeats=1)
    v0 = np.random.default_rng(598).normal(size=p0.free.size)
    lam, V = eigsh(p0.K_free, k=30, sigma=0.0, which="LM", v0=v0, tol=1e-10)
    order = np.argsort(lam)
    lam, V = lam[order], V[:, order]
    b = p0.force[p0.free]
    comp_share = (V.T @ b) ** 2 / lam / float(b @ p0.u_ref[p0.free])
    eig_rows = []
    for key in ("oc0_adam_s0", "oc0_sup_s0", "oc0_lbfgs_s0"):
        e = (fields[key] - p0.u_ref)[p0.free]
        c = V.T @ e
        eig_rows.append({"case": key,
                         "share_of_error_norm_in_mode_1": float(c[0] ** 2 / (e @ e)),
                         "share_in_lowest_5": float((c[:5] ** 2).sum() / (e @ e)),
                         "share_in_lowest_30": float((c ** 2).sum() / (e @ e))})
    summary["eigen_projection_uniform"] = {"lambda": lam[:5].tolist(),
                                           "compliance_share_mode_1": float(comp_share[0]),
                                           "compliance_share_lowest_5": float(comp_share[:5].sum()),
                                           "rows": eig_rows}

    # ---------------- Figure 5: cost and fix (P5, Path C) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), layout="constrained", width_ratios=(1.5, 1))
    ax = axes[0]
    for seed in (0, 1, 2):
        for method, color in (("adam", C_ADAM), ("lbfgs", C_LBFGS)):
            h = history(res, f"oc90_{method}_s{seed}")
            ax.loglog(h["seconds"][1:], h["rel_err_struct"][1:], color=color, lw=1.6 if seed == 0 else 0.9,
                      alpha=1 if seed == 0 else 0.6,
                      label=(("Adam, 20k steps" if method == "adam" else "Adam 10k, then L-BFGS 500")
                             + " (3 seeds)") if seed == 0 else None)
    h = history(res, "oc90_adam40k_s0")
    ax.loglog(h["seconds"][1:], h["rel_err_struct"][1:], color=C_ADAM, ls="--", lw=1.2,
              label="Adam, 40k steps (more time than L-BFGS)")
    t_fe = ref_by["oc90"]["solve_seconds_median"]
    ax.axvline(t_fe, color="k", lw=1)
    ax.text(t_fe * 1.15, 0.5, f"FE direct solve\n{t_fe * 1e3:.0f} ms, residual ~1e-12", fontsize=8)
    ax.set(xlabel="Wall-clock time (s, one CPU core)", ylabel="Relative displacement error (structure)",
           title="(a) Accuracy vs. time on the final design", xlim=(1e-2, 1e3), ylim=(1e-3, 2))
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    names = ["FE direct\n(P1)", "PINN\nAdam", "PINN\nAdam 40k", "PINN\nAdam+L-BFGS"]
    secs = [t_fe, runs["oc90_adam_s0"]["train_seconds"], runs["oc90_adam40k_s0"]["train_seconds"],
            runs["oc90_lbfgs_s0"]["train_seconds"]]
    errs = [None, runs["oc90_adam_s0"]["rel_err_struct"], runs["oc90_adam40k_s0"]["rel_err_struct"],
            runs["oc90_lbfgs_s0"]["rel_err_struct"]]
    bars = ax.bar(names, secs, color=["0.3", C_ADAM, C_ADAM, C_LBFGS], width=0.6)
    for b_, s, e in zip(bars, secs, errs):
        txt = f"{s:.2f} s" if s < 1 else f"{s:.0f} s"
        if e is not None:
            txt += f"\nerr {e:.2%}"
        ax.text(b_.get_x() + b_.get_width() / 2, s * 1.4, txt, ha="center", fontsize=8)
    ax.set(yscale="log", ylim=(1e-2, 3e4), ylabel="Seconds per design (one core)",
           title="(b) Cost of one equilibrium solve")
    fig.savefig(figdir / "cost_and_fix.png", dpi=170)
    plt.close(fig)

    # ---------------- summary numbers used in the report ----------------
    def stats(method):
        vals = [runs[f"oc90_{method}_s{s}"] for s in (0, 1, 2)]
        out = {}
        for key in ("rel_err_struct", "rel_compliance_err", "train_seconds"):
            v = np.array([r[key] for r in vals])
            out[key] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
        return out
    summary["seeds_oc90"] = {"adam": stats("adam"), "lbfgs": stats("lbfgs")}
    t_tr = runs["oc90_lbfgs_s0"]["train_seconds"]
    t_inf = runs["oc90_lbfgs_s0"]["inference_seconds"]
    summary["cost"] = {
        "fe_solve_seconds_median": t_fe, "pinn_inference_seconds": t_inf,
        "pinn_adam_train_seconds": runs["oc90_adam_s0"]["train_seconds"],
        "pinn_lbfgs_train_seconds": t_tr,
        "per_design_cost_ratio_lbfgs": t_tr / t_fe,
        "hypothetical_operator_breakeven_N_if_training_equals_one_pinn": t_tr / (t_fe - t_inf),
        "max_training_seconds_to_break_even_within_90_solves": 90 * (t_fe - t_inf),
    }
    summary["table_runs"] = {k: {kk: runs[k][kk] for kk in ("rel_err_struct", "rel_err_all",
                                                             "rel_compliance_err", "energy_gap",
                                                             "train_seconds")}
                             for k in sorted(runs)}
    # ---------------- checks (reference, energy identity, one error by hand) -------------
    K = pc.full_stiffness(prob)
    checks = {}
    oc90 = next(r for r in ref if r["case"] == "oc90")
    p1 = json.loads((PROJECT.parent / "01_problem_formulation/results/summary.json").read_text())
    checks["p1_compliance_reproduced"] = {"project1": p1["final_compliance"],
                                          "project3_reference": oc90["compliance"],
                                          "fe_relative_residual": oc90["relative_residual"]}
    assert abs(oc90["compliance"] / p1["final_compliance"] - 1) < 1e-9
    for key in ("oc90_adam_s0", "oc90_lbfgs_s0"):
        u = fields[key]
        e = u - prob.u_ref
        pi = 0.5 * u @ (K @ u) - prob.force @ u
        pi_star = -0.5 * prob.compliance_ref
        lhs, rhs = pi - pi_star, 0.5 * e @ (K @ e)
        assert np.all(u[prob.fixed] == 0.0), "hard BCs must hold exactly"
        assert abs(lhs - rhs) <= 1e-9 * abs(pi_star), "Pi - Pi* = e^T K e / 2"
        # Load is -1 on dof 1 (vertical, top-left node): compliance = -u[1].
        comp = -u[1]
        hand = abs(comp - prob.compliance_ref) / prob.compliance_ref
        assert abs(hand - runs[key]["rel_compliance_err"]) < 1e-9
        checks[key] = {"Pi_minus_Pi_star": float(lhs), "half_eKe": float(rhs),
                       "load_node_uy": float(u[1]), "reference_load_node_uy": float(prob.u_ref[1]),
                       "compliance_error_by_hand": float(hand)}
    # float32 training energy agrees with float64 at the FE solution (cancellation fix)
    import torch
    prec = {}
    for name, ny, lw in (("final_120x40", 40, None), ("uniform_240x80_spread_load", 80, 10.0)):
        pp = prob if ny == 40 else pc.build_problem(240, 80, solve_repeats=1, load_width=lw)
        vals = []
        for dt in (torch.float32, torch.float64):
            pc.set_precision(dt)
            s, _, _, _ = pc.TorchEnergy(pp).terms(torch.as_tensor(pp.u_ref, dtype=dt))
            vals.append(float(s))
        pc.set_precision(torch.float32)
        prec[name] = abs(vals[0] / vals[1] - 1)
        assert prec[name] < 1e-5, "float32 strain energy must match float64"
    checks["float32_strain_energy_relative_error_at_fe_solution"] = prec
    summary["checks"] = checks
    (res / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("seeds_oc90", "cost", "spectrum", "eigen_projection_uniform", "checks")}, indent=2))


if __name__ == "__main__":
    main()
