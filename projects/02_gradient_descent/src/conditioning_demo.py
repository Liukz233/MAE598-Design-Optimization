"""Pilot experiments for Project 2; reuse the unchanged Project 1 FE model.

Run from any working directory. Outputs stay inside Project 2 by default.
The default run measures the original 120x40 layout as well as a mesh sweep.
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

# Set before importing numerical libraries; makes small sparse runs predictable.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import cg, eigsh, spsolve

PROJECT = Path(__file__).resolve().parents[1]
P1 = PROJECT.parent / "01_problem_formulation"
sys.path.insert(0, str(P1 / "src"))
from top88 import (  # noqa: E402
    Top88Settings,
    assemble_stiffness,
    element_dofs,
    element_stiffness,
    half_mbb_load_and_supports,
)


def build_system(nx, ny, density=None, emin=1e-9):
    """Square Q4 mesh of the same 3:1 normalized physical domain.

    In 2D with fixed thickness, uniform element-side scaling cancels between
    B^T C B (h^-2) and area (h^2), so the P1 square KE remains valid.
    Nodal unit force and original point support are retained exactly.
    """
    rho = np.full(nx * ny, 0.5) if density is None else np.asarray(density)
    if rho.shape != (nx * ny,):
        raise ValueError("Density must have nx*ny entries in P1 element order")
    settings = Top88Settings(nelx=nx, nely=ny, young_void=emin)
    force, free = half_mbb_load_and_supports(nx, ny)
    full = assemble_stiffness(
        element_dofs(nx, ny), element_stiffness(settings.poisson),
        rho, settings, force.size,
    )
    return full[free, :][:, free].tocsr(), force[free]


def spectrum(A):
    """SPD spectral endpoints and independently evaluated Ritz residuals."""
    start = time.perf_counter()
    n = A.shape[0]
    v0 = np.random.default_rng(598).normal(size=n)
    lo, vlo = eigsh(A, k=1, sigma=0.0, which="LM", v0=v0, tol=1e-10)
    hi, vhi = eigsh(A, k=1, which="LA", v0=v0, tol=1e-10)
    if lo[0] <= 0 or hi[0] <= 0:
        raise ValueError("The supported stiffness must be positive definite")
    residuals = [
        np.linalg.norm(A @ v[:, 0] - lam[0] * v[:, 0]) / hi[0]
        for lam, v in ((lo, vlo), (hi, vhi))
    ]
    return {
        "lambda_min": float(lo[0]), "lambda_max": float(hi[0]),
        "kappa": float(hi[0] / lo[0]),
        "min_eigen_backward_residual": float(residuals[0]),
        "max_eigen_backward_residual": float(residuals[1]),
        "spectrum_seconds": time.perf_counter() - start,
    }


def scaled_system(A):
    diagonal = A.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("Jacobi scaling requires positive diagonal entries")
    S = diags(1.0 / np.sqrt(diagonal))
    return (S @ A @ S).tocsr()


def solve(A, b, method, spectral, scaled_spectral, reference, tol, maxit):
    """Every method is judged by the true, unscaled physical residual.

    Timings include diagnostic work, but exclude assembly, eigenanalysis,
    and reference solution. They are pilot timings, not optimized benchmarks.
    """
    normb = np.linalg.norm(b)
    if normb == 0:
        raise ValueError("The benchmark requires a nonzero load")
    reference_energy = float(reference @ (A @ reference))
    history = []
    start = time.perf_counter()

    def record(k, x, residual=None, force=False):
        residual = np.linalg.norm(A @ x - b) / normb if residual is None else residual
        if force or k <= 20 or k % max(1, k // 250) == 0:
            err = x - reference
            gap = float(err @ (A @ err)) / reference_energy
            history.append({"iteration": k, "relative_residual": float(residual),
                            "relative_energy_gap": max(gap, 0.0)})
        return residual

    x = np.zeros_like(b)
    record(0, x, 1.0, force=True)
    alpha = None
    iterations = 0
    if method in ("GD", "Jacobi-GD"):
        spec = spectral if method == "GD" else scaled_spectral
        alpha = 2.0 / (spec["lambda_min"] + spec["lambda_max"])
        invdiag = np.ones_like(b) if method == "GD" else 1.0 / A.diagonal()
        residual = 1.0
        for k in range(1, maxit + 1):
            x += alpha * invdiag * (b - A @ x)
            residual = float(np.linalg.norm(A @ x - b) / normb)
            record(k, x, residual)
            iterations = k
            if residual <= tol:
                break
    else:
        M = None if method == "CG" else diags(1.0 / A.diagonal())

        def callback(current):
            nonlocal iterations
            iterations += 1
            record(iterations, current)

        x, info = cg(A, b, x0=x, M=M, rtol=tol, atol=0.0,
                     maxiter=maxit, callback=callback)
        if info < 0:
            raise RuntimeError(f"CG breakdown: {info}")
    residual = float(np.linalg.norm(A @ x - b) / normb)
    record(iterations, x, residual, force=True)
    # Deduplicate final sample when already recorded by the callback.
    history = list({row["iteration"]: row for row in history}.values())
    err = x - reference
    return {
        "method": method, "iterations": iterations, "max_iterations": maxit,
        "converged": bool(residual <= tol), "relative_residual": residual,
        "relative_displacement_error": float(np.linalg.norm(err) / np.linalg.norm(reference)),
        "relative_energy_gap": max(float(err @ (A @ err)) / reference_energy, 0.0),
        "relative_compliance_error": float(abs(b @ x - b @ reference) / abs(b @ reference)),
        "seconds_with_diagnostics": time.perf_counter() - start,
        "step": alpha,
    }, history


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_small_case():
    """Hand-solvable FE slice, plus dense/sparse spectral agreement."""
    ke = element_stiffness(0.3)
    full = assemble_stiffness(element_dofs(1, 1), ke, np.ones(1),
                              Top88Settings(nelx=1, nely=1), 8)
    # Only the two top-node vertical DOFs are free for this helper check.
    A = full[[1, 5], :][:, [1, 5]].toarray()
    expected = np.array([[45, 5], [5, 45]]) / 91.0
    np.testing.assert_allclose(A, expected, rtol=1e-13, atol=1e-15)
    ev = np.linalg.eigvalsh(A)
    np.testing.assert_allclose(ev, np.array([40, 50]) / 91.0, rtol=1e-13)
    small, b = build_system(6, 2)
    dense = np.linalg.eigvalsh(small.toarray())
    sparse = spectrum(small)
    np.testing.assert_allclose([sparse["lambda_min"], sparse["lambda_max"]],
                               dense[[0, -1]], rtol=1e-9)
    u = spsolve(small, b)
    # Directional derivative independently checks gradient and energy identity.
    d = np.random.default_rng(598).normal(size=b.size)
    x = 0.37 * u
    eps = 1e-3
    energy = lambda v: 0.5 * v @ (small @ v) - b @ v
    derivative = (energy(x + eps*d) - energy(x - eps*d)) / (2*eps)
    np.testing.assert_allclose(derivative, d @ (small @ x - b), rtol=1e-7, atol=1e-8)
    gap = energy(x) - energy(u)
    np.testing.assert_allclose(gap, 0.5 * (x-u) @ (small @ (x-u)), rtol=1e-10)
    return {"one_element_matrix": A.tolist(), "exact_eigenvalues": [40/91, 50/91],
            "exact_condition_number": 1.25, "dense_sparse_check": "passed",
            "energy_gradient_check": "passed", "energy_gap_check": "passed"}


def make_figures(output, condition_rows, solver_rows, histories, spectra):
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    mesh = [r for r in condition_rows if r["layout"] == "uniform"]
    fig, ax = plt.subplots(figsize=(6.5, 4.1), layout="constrained")
    ax.loglog([r["nely"] for r in mesh], [r["kappa"] for r in mesh], "o-", label="Original")
    ax.loglog([r["nely"] for r in mesh], [r["jacobi_kappa"] for r in mesh], "s--", label="Jacobi scaled")
    ax.set(xlabel="Elements through beam height", ylabel="Condition number",
           title="Mesh refinement: uniform half MBB beam")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    fig.savefig(figdir / "mesh_conditioning.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for name, ev, scaled_ev in spectra:
        axes[0].semilogy(np.arange(1, len(ev)+1)/len(ev), ev, label=name)
        axes[1].semilogy(np.arange(1, len(ev)+1)/len(ev), scaled_ev, label=name)
    for ax, title in zip(axes, ["Original stiffness", "Jacobi-scaled stiffness"]):
        ax.set(xlabel="Normalized eigenvalue index", ylabel="Eigenvalue", title=title)
        ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(figdir / "eigenvalue_spectra.png", dpi=170); plt.close(fig)

    for case in ["uniform_24x8", "optimized_120x40_Emin_1e-09"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        uniform_case = case.startswith("uniform")
        metrics = ("relative_residual", "relative_residual" if uniform_case else "relative_energy_gap")
        max_iteration = 0
        for method in ("GD", "Jacobi-GD", "CG", "Jacobi-PCG"):
            rows = histories.get((case, method))
            if rows is None:
                continue
            max_iteration = max(max_iteration, rows[-1]["iteration"])
            for ax, metric in zip(axes, metrics):
                ax.semilogy([r["iteration"] for r in rows],
                            [max(r[metric], 1e-30) for r in rows], label=method)
        axes[0].axhline(1e-6, color="gray", ls=":", label="Residual tolerance")
        if uniform_case:
            axes[1].axhline(1e-6, color="gray", ls=":", label="Residual tolerance")
        labels = ("Relative physical residual", "Relative physical residual" if uniform_case else "Relative energy gap")
        for ax, ylabel in zip(axes, labels):
            ax.set(xlabel="Iteration", ylabel=ylabel, xlim=(0, max_iteration*1.02))
            ax.legend(); ax.grid(alpha=0.2)
        if uniform_case:
            axes[0].set_title("Full iteration budget")
            axes[1].set(xlim=(0, 500), title="First 500 iterations")
        fig.savefig(figdir / f"convergence_{case}.png", dpi=170); plt.close(fig)

    contrast = [r for r in condition_rows if r["layout"] == "optimized"]
    fig, ax = plt.subplots(figsize=(6.5, 4.1), layout="constrained")
    ax.loglog([1/r["emin"] for r in contrast], [r["kappa"] for r in contrast], "o-", label="Original")
    ax.loglog([1/r["emin"] for r in contrast], [r["jacobi_kappa"] for r in contrast], "s--", label="Jacobi scaled")
    ax.set(xlabel="Solid-to-void modulus ratio", ylabel="Condition number",
           title="Fixed Project 1 topology: material contrast")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    fig.savefig(figdir / "material_contrast.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT)
    parser.add_argument("--gd-maxit", type=int, default=120000)
    parser.add_argument("--cg-maxit", type=int, default=5000)
    args = parser.parse_args()
    output = args.output.resolve()
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    verification = verify_small_case()
    density_path = P1 / "results" / "final_density.csv"
    rho = np.loadtxt(density_path, delimiter=",")
    if rho.shape != (40, 120):
        raise ValueError(f"Unexpected P1 density shape {rho.shape}")
    rho = rho.ravel(order="F")
    condition_rows, solver_rows, spectra = [], [], []
    histories = {}
    cases = [(f"uniform_{nx}x{ny}", nx, ny, None, 1e-9)
             for nx, ny in ((12, 4), (24, 8), (48, 16), (120, 40))]
    cases += [(f"optimized_120x40_Emin_{emin:.0e}", 120, 40, rho, emin)
              for emin in (1e-3, 1e-6, 1e-9)]
    for name, nx, ny, density, emin in cases:
        print(f"Analyzing {name}", flush=True)
        A, b = build_system(nx, ny, density, emin)
        J = scaled_system(A)
        raw, scaled = spectrum(A), spectrum(J)
        start = time.perf_counter()
        u = spsolve(A, b)
        direct_time = time.perf_counter() - start
        direct_residual = float(np.linalg.norm(A @ u-b)/np.linalg.norm(b))
        row = {"case": name, "layout": "uniform" if density is None else "optimized",
               "nelx": nx, "nely": ny, "free_dofs": b.size, "emin": emin,
               **raw, "jacobi_kappa": scaled["kappa"],
               "jacobi_lambda_min": scaled["lambda_min"],
               "jacobi_lambda_max": scaled["lambda_max"],
               "jacobi_min_eigen_backward_residual": scaled["min_eigen_backward_residual"],
               "jacobi_max_eigen_backward_residual": scaled["max_eigen_backward_residual"],
               "reference_compliance": float(b @ u),
               "reference_residual": direct_residual,
               "reference_solve_seconds": direct_time}
        condition_rows.append(row)
        if density is None:
            _, free = half_mbb_load_and_supports(nx, ny)
            x_coordinates = np.repeat(np.linspace(0.0, 120.0, nx+1), ny+1)
            trial = np.zeros(2 * len(x_coordinates))
            trial[1::2] = 1.0 - x_coordinates / 120.0
            trial = trial[free]
            effective_E = emin + 0.5**3 * (1.0 - emin)
            affine_energy = float(trial @ (A @ trial))
            np.testing.assert_allclose(affine_energy, effective_E/(6*1.3), rtol=1e-9)
            diagonal_ratio = float(A.diagonal().max()/A.diagonal().min())
            np.testing.assert_allclose(diagonal_ratio, 4.0, rtol=1e-13)
            assert scaled["kappa"] >= raw["kappa"]/diagonal_ratio * (1-1e-8)
            verification[name] = {"diagonal_ratio": diagonal_ratio,
                                  "affine_trial_energy": affine_energy,
                                  "jacobi_lower_bound": raw["kappa"]/diagonal_ratio}
        if density is None and nx <= 24:
            ev, jev = np.linalg.eigvalsh(A.toarray()), np.linalg.eigvalsh(J.toarray())
            spectra.append((f"{nx} x {ny}", ev, jev))
            csv_write(results / f"spectrum_{name}.csv",
                      [{"index": i+1, "eigenvalue": float(e), "jacobi_eigenvalue": float(j)}
                       for i, (e, j) in enumerate(zip(ev, jev))])
        if density is not None and emin == 1e-9:
            expected = json.loads((P1 / "results" / "summary.json").read_text())["final_compliance"]
            np.testing.assert_allclose(b @ u, expected, rtol=1e-7)
            verification["project1_compliance_reproduced"] = float(b @ u)
        methods = ["CG", "Jacobi-PCG"]
        if density is None and nx <= 48:
            methods = ["GD", "Jacobi-GD"] + methods
        for method in methods:
            cap = args.gd_maxit if "GD" in method else args.cg_maxit
            metrics, history = solve(A, b, method, raw, scaled, u, 1e-6, cap)
            solver_rows.append({"case": name, **metrics})
            histories[(name, method)] = history
            csv_write(results / f"history_{name}_{method}.csv", history)
            print(f"  {method}: {metrics['iterations']} iterations, "
                  f"residual {metrics['relative_residual']:.3e}, "
                  f"converged {metrics['converged']}", flush=True)
        csv_write(results / "conditioning.csv", condition_rows)
        csv_write(results / "solvers.csv", solver_rows)
    make_figures(output, condition_rows, solver_rows, histories, spectra)
    (results / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    metadata = {
        "status": "pilot experiments; not a final submission",
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "seed": 598,
        "blas_threads": 1, "tolerance": 1e-6,
        "gd_maxit": args.gd_maxit, "cg_maxit": args.cg_maxit,
        "density_sha256": hashlib.sha256(density_path.read_bytes()).hexdigest(),
        "project1_source_sha256": hashlib.sha256((P1 / "src" / "top88.py").read_bytes()).hexdigest(),
        "physical_domain": "[0,120] x [0,40], constant unit thickness, h=40/nely",
        "load_support": "P1 unit point force and nodal roller unchanged",
        "timing_note": "Single runs including convergence diagnostics; not speedup benchmarks",
        "spectral_note": "Full spectra only for 12x4 and 24x8; sparse endpoints elsewhere",
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Pilot outputs written to", output, flush=True)


if __name__ == "__main__":
    main()
