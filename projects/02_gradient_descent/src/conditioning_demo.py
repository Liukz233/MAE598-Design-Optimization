"""Reproduce Project 2: conditioning and solution of the half-MBB energy problem.

Run from any working directory. Outputs stay inside Project 2 by default.
The default run measures the original 120x40 layout as well as a mesh sweep.
Gradient descent runs to the tolerance on the three smaller meshes, which
takes about two minutes; the rest of the run takes about half a minute.
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
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, eigsh, spsolve

PROJECT = Path(__file__).resolve().parents[1]
P1 = PROJECT.parent / "01_problem_formulation"
sys.path.insert(0, str(P1 / "src"))
from top88 import (  # noqa: E402
    Top88Settings,
    assemble_stiffness,
    element_dofs,
    element_stiffness,
    finite_element_analysis,
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


def spectrum(A, return_vectors=False):
    """SPD spectral endpoints and independently evaluated Ritz residuals.

    With return_vectors=True, also return the unit eigenvectors of
    lambda_min and lambda_max.
    """
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
    stats = {
        "lambda_min": float(lo[0]), "lambda_max": float(hi[0]),
        "kappa": float(hi[0] / lo[0]),
        "min_eigen_backward_residual": float(residuals[0]),
        "max_eigen_backward_residual": float(residuals[1]),
        "spectrum_seconds": time.perf_counter() - start,
    }
    return (stats, vlo[:, 0], vhi[:, 0]) if return_vectors else stats


def gd_prediction(spec, softest, stiffest, b, u, tol):
    """Predict fixed-step GD updates from the softest mode alone.

    Once the other components have decayed, the residual is c1*q**k, where
    c1 = |v1.F|/|F| is the load's share on the softest mode and
    q = (kappa-1)/(kappa+1) is the per-update factor of the optimal step.
    The stiffest mode decays by -q per update, but the load barely excites it.
    """
    c1 = abs(softest @ b) / np.linalg.norm(b)
    q = (spec["kappa"] - 1.0) / (spec["kappa"] + 1.0)
    compliance_share = (softest @ b) ** 2 / spec["lambda_min"] / float(b @ u)
    return {"softest_mode_load_share": float(c1),
            "stiffest_mode_load_share": float(abs(stiffest @ b) / np.linalg.norm(b)),
            "softest_mode_compliance_share": float(compliance_share),
            "gd_predicted_iterations": int(np.ceil(np.log(tol / c1) / np.log(q)))}


def scaled_system(A):
    diagonal = A.diagonal()
    if np.any(diagonal <= 0):
        raise ValueError("Jacobi scaling requires positive diagonal entries")
    S = diags(1.0 / np.sqrt(diagonal))
    return (S @ A @ S).tocsr()


def solve(A, b, method, spectral, scaled_spectral, reference, tol, maxit):
    """Every method is judged by the true, unscaled physical residual.

    Timings include diagnostic work, but exclude assembly, eigenanalysis,
    and reference solution. The report compares iterations at a common tolerance.
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
    }, history, x


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
    check_A = csr_matrix(A)
    check_b = np.array([1.0, 0.0])
    check_u = np.linalg.solve(A, check_b)
    check_spec = {"lambda_min": 40/91, "lambda_max": 50/91}
    gd, gd_history, _ = solve(check_A, check_b, "GD", check_spec, check_spec,
                           check_u, 1e-6, 20)
    cg_result, _, _ = solve(check_A, check_b, "CG", check_spec, check_spec,
                         check_u, 1e-6, 20)
    assert gd["converged"] and gd["iterations"] == 7
    assert cg_result["converged"] and cg_result["iterations"] == 2
    np.testing.assert_allclose(
        [r["relative_residual"] for r in gd_history],
        [9.0**(-r["iteration"]) for r in gd_history], rtol=1e-8, atol=1e-15,
    )
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
            "exact_condition_number": 1.25,
            "hand_check_gd_updates": gd["iterations"],
            "hand_check_cg_updates": cg_result["iterations"],
            "hand_check_gd_residual_formula": "9**(-k)",
            "dense_sparse_check": "passed",
            "energy_gradient_check": "passed", "energy_gap_check": "passed"}


def make_figures(output, condition_rows, solver_rows, histories, spectra):
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    mesh = [r for r in condition_rows if r["layout"] == "uniform"]
    ny = np.array([r["nely"] for r in mesh], dtype=float)
    kappa = np.array([r["kappa"] for r in mesh])
    fig, ax = plt.subplots(figsize=(6.5, 4.1), layout="constrained")
    ax.loglog(ny, kappa, "o-", label="Original")
    ax.loglog(ny, [r["jacobi_kappa"] for r in mesh], "s--", label="Jacobi scaled")
    ax.loglog(ny, kappa[0] * (ny / ny[0])**2, ":", color="0.45", label=r"Slope 2 ($\kappa\propto n_y^2$)")
    ax.set_xticks(ny, [str(int(n)) for n in ny])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlabel=r"Elements through beam height $n_y$", ylabel="Condition number",
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

    # D3: GD versus CG on the 24x8 mesh, with the one-mode GD prediction.
    case = "uniform_24x8"
    row = next(r for r in condition_rows if r["case"] == case)
    gd, cgh = histories[(case, "GD")], histories[(case, "CG")]
    colors = {"GD": "#1f77b4", "CG": "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    ax = axes[0]
    for method, rows in (("GD", gd), ("CG", cgh)):
        ax.semilogy([r["iteration"] for r in rows],
                    [max(r["relative_residual"], 1e-30) for r in rows],
                    color=colors[method], label=method)
    q = (row["kappa"] - 1.0) / (row["kappa"] + 1.0)
    k = np.linspace(2000, gd[-1]["iteration"], 200)
    ax.semilogy(k, row["softest_mode_load_share"] * q**k, "k:", lw=1.6,
                label=r"Predicted $c_1q^k$")
    ax.axhline(1e-6, color="gray", ls="--", lw=0.8, label="Tolerance")
    ax.set(xlabel="Iteration", ylabel="Relative residual",
           title="(a) Full run", xlim=(0, gd[-1]["iteration"] * 1.02))
    ax.legend(); ax.grid(alpha=0.2)
    ax = axes[1]
    for method, rows in (("GD", gd), ("CG", cgh)):
        rows = [r for r in rows if r["iteration"] <= 500]
        its = [r["iteration"] for r in rows]
        ax.semilogy(its, [max(r["relative_residual"], 1e-30) for r in rows],
                    color=colors[method], label=f"{method}: residual")
        ax.semilogy(its, [max(r["relative_energy_gap"], 1e-30) for r in rows],
                    color=colors[method], ls="--", label=f"{method}: energy gap")
    ax.axhline(1e-6, color="gray", ls="--", lw=0.8)
    ax.set(xlabel="Iteration", ylabel="Relative residual or energy gap",
           title="(b) First 500 iterations", xlim=(0, 500), ylim=(1e-15, 10))
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.2)
    fig.savefig(figdir / f"convergence_{case}.png", dpi=170); plt.close(fig)

    # D4 for material contrast: condition numbers and CG versus Jacobi-PCG.
    contrast = [r for r in condition_rows if r["layout"] == "optimized"]
    uniform_fine = next(r for r in condition_rows if r["case"] == "uniform_120x40")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), layout="constrained")
    ax = axes[0]
    ax.loglog([1/r["emin"] for r in contrast], [r["kappa"] for r in contrast], "o-", label="Original")
    ax.loglog([1/r["emin"] for r in contrast], [r["jacobi_kappa"] for r in contrast], "s--",
              label="Jacobi scaled")
    ax.axhline(uniform_fine["jacobi_kappa"], color="0.45", ls=":",
               label="Uniform 120×40, Jacobi scaled")
    ax.set(xlabel=r"Solid-to-void modulus ratio $E_0/E_{\min}$", ylabel="Condition number",
           title="(a) Project 1 layout: material contrast")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    ax = axes[1]
    case = "optimized_120x40_Emin_1e-09"
    for method, color in (("CG", "#ba5a38"), ("Jacobi-PCG", "#167d8d")):
        rows = histories[(case, method)]
        ax.semilogy([r["iteration"] for r in rows],
                    [max(r["relative_residual"], 1e-30) for r in rows], color=color, label=method)
    ax.axhline(1e-6, color="gray", ls="--", lw=0.8, label="Tolerance")
    ax.set(xlabel="Iteration", ylabel="Relative residual",
           title=r"(b) Project 1 layout, $E_{\min}=10^{-9}$")
    ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(figdir / "optimized_jacobi_fix.png", dpi=170); plt.close(fig)


def plot_extreme_modes(output, nx=24, ny=8):
    """Draw the softest and stiffest eigenvectors as deformed meshes."""
    A, b = build_system(nx, ny)
    w, V = np.linalg.eigh(A.toarray())
    force, free = half_mbb_load_and_supports(nx, ny)
    h = 40.0 / ny
    X, Y = np.meshgrid(np.arange(nx + 1) * h, 40.0 - np.arange(ny + 1) * h)
    u = spsolve(A, b)
    share = (V[:, 0] @ b) ** 2 / w[0] / float(b @ u)
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), layout="constrained")
    mantissa, exponent = f"{w[0]:.1e}".split("e")
    titles = (f"(a) Softest mode: $\\lambda_{{\\min}}={mantissa}\\times10^{{{int(exponent)}}}$\n"
              f"the whole beam deflects; {100*share:.0f}% of the compliance",
              f"(b) Stiffest mode: $\\lambda_{{\\max}}={w[-1]:.2f}$\n"
              "neighboring nodes move in opposite directions")
    for ax, idx, title in zip(axes, (0, -1), titles):
        vector = V[:, idx] * (np.sign(V[:, idx] @ b) or 1.0)
        full = np.zeros(force.size)
        full[free] = vector
        ux = full[0::2].reshape((ny + 1, nx + 1), order="F")
        uy = full[1::2].reshape((ny + 1, nx + 1), order="F")  # positive is upward
        scale = (1.2 if idx == 0 else 0.6) * h / np.abs(full).max()
        ax.plot(X, Y, color="0.85", lw=0.5); ax.plot(X.T, Y.T, color="0.85", lw=0.5)
        Xd, Yd = X + scale * ux, Y + scale * uy
        ax.plot(Xd, Yd, color="#173a5e", lw=0.8); ax.plot(Xd.T, Yd.T, color="#173a5e", lw=0.8)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)
    fig.savefig(output / "figures" / "extreme_modes.png", dpi=170)
    plt.close(fig)
    return {"mesh": f"{nx}x{ny}", "lambda_1": float(w[0]), "lambda_2": float(w[1]),
            "lambda_max": float(w[-1]), "softest_mode_compliance_share": float(share)}


def median_seconds(fn, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def project_comparison(output, rho, A, b, pcg_solution, pcg_metrics):
    """Compare the P1 direct FE solution and P2 solvers on one fixed layout."""
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    settings = Top88Settings(nelx=120, nely=40)
    force, free = half_mbb_load_and_supports(120, 40)
    edof = element_dofs(120, 40)
    p1_full, _, p1_compliance = finite_element_analysis(
        rho, settings, edof, element_stiffness(0.3), force, free,
    )
    p1 = p1_full[free]
    expected = json.loads((P1 / "results" / "summary.json").read_text())["final_compliance"]
    np.testing.assert_allclose(p1_compliance, expected, rtol=1e-8)
    budget = pcg_metrics["iterations"]
    cg_equal, info = cg(A, b, rtol=1e-6, atol=0.0, maxiter=budget)
    assert info == budget, "Update comparison if CG reaches tolerance within the shared budget"
    # "Structure" DOFs touch at least one element with density >= 0.5.
    on_structure = np.zeros(force.size, dtype=bool)
    on_structure[np.unique(edof[rho >= 0.5])] = True
    on_structure = on_structure[free]
    jacobi = diags(1.0 / A.diagonal())
    seconds = {
        "Project 1 direct": median_seconds(lambda: spsolve(A, b)),
        "Project 2 CG": median_seconds(lambda: cg(A, b, rtol=1e-6, atol=0.0, maxiter=budget)),
        "Project 2 Jacobi-PCG": median_seconds(
            lambda: cg(A, b, rtol=1e-6, atol=0.0, maxiter=5000, M=jacobi)),
    }
    rows = []
    for name, x, count in (("Project 1 direct", p1, None),
                           ("Project 2 CG", cg_equal, budget),
                           ("Project 2 Jacobi-PCG", pcg_solution, budget)):
        err = x - p1
        rows.append({"method": name, "iterations": count,
                     "relative_residual": float(np.linalg.norm(A @ x-b)/np.linalg.norm(b)),
                     "compliance": float(b @ x),
                     "relative_displacement_error_vs_project1": float(np.linalg.norm(err)/np.linalg.norm(p1)),
                     "relative_displacement_error_on_structure": float(
                         np.linalg.norm(err[on_structure])/np.linalg.norm(p1[on_structure])),
                     "relative_displacement_error_in_void": float(
                         np.linalg.norm(err[~on_structure])/np.linalg.norm(p1[~on_structure])),
                     "relative_compliance_error_vs_project1": float(abs(b @ x-b @ p1)/abs(b @ p1)),
                     "median_solve_seconds": seconds[name]})
    assert rows[-1]["relative_residual"] <= 1e-6
    assert rows[-1]["relative_displacement_error_on_structure"] < 1e-6
    assert rows[1]["compliance"] < rows[0]["compliance"], "Truncated CG should underestimate compliance"
    csv_write(output / "results" / "project1_project2_comparison.csv", rows)

    def vertical_field(solution):
        full = np.zeros_like(force)
        full[free] = solution
        return full[1::2].reshape((41, 121), order="F")

    # Average the four nodal values onto each element, then mask only the plot.
    # All densities and DOFs remain in the original SIMP equilibrium solve.
    visible = rho.reshape((40, 120), order="F") >= 0.5
    def material_field(nodal):
        values = (nodal[:-1, :-1] + nodal[1:, :-1]
                  + nodal[:-1, 1:] + nodal[1:, 1:]) / 4.0
        return np.ma.array(values, mask=~visible)
    p1_y = material_field(vertical_field(p1))
    cg_y = material_field(vertical_field(cg_equal))
    pcg_y = material_field(vertical_field(pcg_solution))
    assert p1_y.count() == np.count_nonzero(visible)
    scale = np.max(np.abs(p1_y))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
    im = axes[0, 0].imshow(-p1_y/scale, extent=(0, 120, 0, 40), origin="upper", cmap="viridis")
    axes[0, 0].set(title="(a) Project 1 direct solve (reference)", xlabel="x", ylabel="y")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.7, label="Normalized downward displacement")
    norm = LogNorm(vmin=1e-11, vmax=1e-1)
    for ax, values, label in ((axes[0, 1], cg_y, f"(b) CG after {budget:,} updates: error"),
                              (axes[1, 0], pcg_y, f"(c) Jacobi-PCG after {budget:,} updates: error")):
        err = np.ma.array(np.maximum(np.abs(values - p1_y) / scale, 1e-16), mask=~visible)
        im = ax.imshow(err, extent=(0, 120, 0, 40), origin="upper", cmap="magma", norm=norm)
        ax.set(title=label, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, shrink=0.7, label="|error| / max displacement")
    ax = axes[1, 1]
    values = [r["relative_residual"] for r in rows[1:]]
    ax.bar(["CG", "Jacobi-PCG"], values, color=["#ba5a38", "#167d8d"], width=0.55)
    for x, value in enumerate(values):
        ax.text(x, value*1.8, f"{value:.2e}", ha="center", fontsize=10)
    ax.axhline(1e-6, color="#555555", ls="--", label="Tolerance: 1e-6")
    ax.set(yscale="log", ylim=(1e-8, 1), ylabel="Relative equilibrium residual",
           title=f"(d) Residual after {budget:,} updates")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Optimized beam, same budget for both solvers (elements with density ≥ 0.5 shown)",
                 fontsize=13)
    fig.savefig(figdir / "project1_project2_comparison.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 3.6), facecolor="#f6f8fa")
    fig.text(0.045, 0.9, "PROJECT 2  /  ILL-CONDITIONED OPTIMIZATION", color="#173a5e", fontsize=16, weight="bold")
    fig.text(0.045, 0.81, "From an optimized beam to a reliable equilibrium solver", color="#303e4a", fontsize=12)
    ax = fig.add_axes([0.045, 0.24, 0.39, 0.44])
    ax.imshow(rho.reshape((40, 120), order="F"), cmap="gray_r", vmin=0, vmax=1)
    ax.axis("off"); ax.set_title("Project 1: material layout", loc="left", fontsize=11)
    ax = fig.add_axes([0.55, 0.25, 0.4, 0.43])
    case = "optimized_120x40_Emin_1e-09"
    for method, color in (("CG", "#ba5a38"), ("Jacobi-PCG", "#167d8d")):
        with (output / "results" / f"history_{case}_{method}.csv").open() as f:
            history = list(csv.DictReader(f))
        ax.semilogy([int(r["iteration"]) for r in history],
                    [float(r["relative_residual"]) for r in history], color=color, label=method, lw=2)
    ax.axhline(1e-6, color="#777777", ls=":")
    ax.set(xlabel="Iteration", ylabel="Residual", title="Project 2: iterative convergence")
    ax.legend(fontsize=9); ax.grid(alpha=0.15)
    fig.text(0.045, 0.09, "OptiForge  |  Kangzheng Liu  |  MAE 598/494 Design Optimization", fontsize=10, color="#536471")
    fig.savefig(figdir / "project2_cover.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT)
    parser.add_argument("--gd-maxit", type=int, default=2_000_000)
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
        (raw, softest, stiffest), scaled = spectrum(A, return_vectors=True), spectrum(J)
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
        if density is None:
            row.update(gd_prediction(raw, softest, stiffest, b, u, 1e-6))
        if nx == 120:
            # CG counts depend on how many small eigenvalues there are, not only kappa.
            low = eigsh(J, k=60, sigma=0.0, which="LM", tol=1e-8, return_eigenvectors=False,
                        v0=np.random.default_rng(598).normal(size=b.size))
            row["jacobi_eigenvalues_below_1e-3"] = int(np.sum(low < 1e-3))
            assert np.max(low) > 1e-3, "Increase k: all computed eigenvalues are below 1e-3"
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
            trial_quotient = affine_energy / float(trial @ trial)
            assert raw["lambda_min"] <= trial_quotient
            diagonal_ratio = float(A.diagonal().max()/A.diagonal().min())
            np.testing.assert_allclose(diagonal_ratio, 4.0, rtol=1e-13)
            assert scaled["kappa"] >= raw["kappa"]/diagonal_ratio * (1-1e-8)
            verification[name] = {"diagonal_ratio": diagonal_ratio,
                                  "affine_trial_energy": affine_energy,
                                  "affine_trial_rayleigh_quotient": trial_quotient,
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
            metrics, history, solution = solve(A, b, method, raw, scaled, u, 1e-6, cap)
            solver_rows.append({"case": name, **metrics})
            histories[(name, method)] = history
            csv_write(results / f"history_{name}_{method}.csv", history)
            print(f"  {method}: {metrics['iterations']} iterations, "
                  f"residual {metrics['relative_residual']:.3e}, "
                  f"converged {metrics['converged']}", flush=True)
            if method == "GD" and metrics["converged"]:
                predicted = row["gd_predicted_iterations"]
                # The one-mode formula ignores the other decaying components.
                assert abs(metrics["iterations"] - predicted) <= 1e-3 * predicted
                verification[name]["gd_iterations_measured"] = metrics["iterations"]
                verification[name]["gd_iterations_predicted"] = predicted
            if density is not None and emin == 1e-9 and method == "Jacobi-PCG":
                project_comparison(output, rho, A, b, solution, metrics)
        csv_write(results / "conditioning.csv", condition_rows)
        csv_write(results / "solvers.csv", solver_rows)
    make_figures(output, condition_rows, solver_rows, histories, spectra)
    verification["extreme_modes"] = plot_extreme_modes(output)
    uniform = [r for r in condition_rows if r["layout"] == "uniform"]
    log_ny = np.log([r["nely"] for r in uniform])
    cg_counts = {r["case"]: r["iterations"] for r in solver_rows if r["method"] == "CG"}
    verification["mesh_scaling"] = {
        "kappa_loglog_slope_vs_ny": float(np.polyfit(log_ny, np.log([r["kappa"] for r in uniform]), 1)[0]),
        "jacobi_kappa_loglog_slope_vs_ny": float(
            np.polyfit(log_ny, np.log([r["jacobi_kappa"] for r in uniform]), 1)[0]),
        "cg_iterations_over_sqrt_kappa": {
            r["case"]: cg_counts[r["case"]] / np.sqrt(r["kappa"]) for r in uniform},
    }
    (results / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    metadata = {
        "experiment": "Project 2 half-MBB conditioning and solver comparison",
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "seed": 598,
        "blas_threads": 1, "tolerance": 1e-6,
        "gd_maxit": args.gd_maxit, "cg_maxit": args.cg_maxit,
        "density_sha256": hashlib.sha256(density_path.read_bytes()).hexdigest(),
        "project1_source_sha256": hashlib.sha256((P1 / "src" / "top88.py").read_bytes()).hexdigest(),
        "physical_domain": "[0,120] x [0,40], constant unit thickness, h=40/nely",
        "load_support": "P1 unit point force and nodal roller unchanged",
        "timing_note": ("Solver timings in solvers.csv include convergence diagnostics; "
                        "project1_project2_comparison.csv has clean median timings"),
        "spectral_note": "Full spectra only for 12x4 and 24x8; sparse endpoints elsewhere",
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Results written to", output, flush=True)


if __name__ == "__main__":
    main()
