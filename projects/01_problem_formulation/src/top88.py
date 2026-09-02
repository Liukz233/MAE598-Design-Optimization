"""Educational Python implementation of the classic 88-line SIMP--OC method.

The default command reproduces the official DTU example::

    top88(120, 40, 0.5, 3.0, 3.5, 1)

The finite-element model, half MBB boundary conditions, SIMP interpolation,
sensitivity/density filters, and optimality-criteria update follow Andreassen
et al. (2011). The implementation is expanded for readability and adds
reproducible CSV, JSON, and PNG outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Top88Settings:
    """Inputs corresponding to the six arguments of the MATLAB ``top88``."""

    nelx: int = 120
    nely: int = 40
    volfrac: float = 0.5
    penal: float = 3.0
    rmin: float = 3.5
    filter_type: int = 1
    tolerance: float = 0.01
    max_iterations: int = 250
    young_solid: float = 1.0
    young_void: float = 1.0e-9
    poisson: float = 0.3
    move_limit: float = 0.2


@dataclass
class Top88Result:
    """Optimized physical density field and convergence information."""

    density: np.ndarray
    history: list[dict[str, float | int]]
    runtime_seconds: float


def element_stiffness(poisson: float = 0.3) -> np.ndarray:
    """Return the 8-by-8 bilinear square-element stiffness matrix."""

    a11 = np.array(
        [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]],
        dtype=float,
    )
    a12 = np.array(
        [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]],
        dtype=float,
    )
    b11 = np.array(
        [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]],
        dtype=float,
    )
    b12 = np.array(
        [[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]],
        dtype=float,
    )
    a = np.block([[a11, a12], [a12.T, a11]])
    b = np.block([[b11, b12], [b12.T, b11]])
    return (a + poisson * b) / (24.0 * (1.0 - poisson**2))


def element_dofs(nelx: int, nely: int) -> np.ndarray:
    """Build the element-to-global degree-of-freedom table."""

    edof = np.empty((nelx * nely, 8), dtype=int)
    for ex in range(nelx):
        for ey in range(nely):
            element = ex * nely + ey
            node_left = (nely + 1) * ex + ey
            node_right = (nely + 1) * (ex + 1) + ey
            edof[element] = [
                2 * node_left + 2,
                2 * node_left + 3,
                2 * node_right + 2,
                2 * node_right + 3,
                2 * node_right,
                2 * node_right + 1,
                2 * node_left,
                2 * node_left + 1,
            ]
    return edof


def filter_matrix(nelx: int, nely: int, rmin: float) -> tuple[csc_matrix, np.ndarray]:
    """Construct the distance-weighted filter matrix used by ``top88``."""

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    radius = int(np.ceil(rmin)) - 1
    for ex in range(nelx):
        for ey in range(nely):
            row = ex * nely + ey
            for nx in range(max(ex - radius, 0), min(ex + radius + 1, nelx)):
                for ny in range(max(ey - radius, 0), min(ey + radius + 1, nely)):
                    weight = rmin - np.hypot(ex - nx, ey - ny)
                    if weight > 0.0:
                        rows.append(row)
                        cols.append(nx * nely + ny)
                        weights.append(float(weight))
    matrix = coo_matrix(
        (weights, (rows, cols)), shape=(nelx * nely, nelx * nely)
    ).tocsc()
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    return matrix, row_sums


def half_mbb_load_and_supports(nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the unit load and free DOFs for the symmetric half MBB beam."""

    ndof = 2 * (nelx + 1) * (nely + 1)
    force = np.zeros(ndof)
    force[1] = -1.0
    symmetry_dofs = np.arange(0, 2 * (nely + 1), 2, dtype=int)
    vertical_support = np.array([ndof - 1], dtype=int)
    fixed = np.union1d(symmetry_dofs, vertical_support)
    free = np.setdiff1d(np.arange(ndof, dtype=int), fixed)
    return force, free


def assemble_stiffness(
    edof: np.ndarray,
    ke: np.ndarray,
    physical_density: np.ndarray,
    settings: Top88Settings,
    ndof: int,
) -> csc_matrix:
    """Assemble the symmetric sparse global stiffness matrix."""

    modulus = settings.young_void + physical_density**settings.penal * (
        settings.young_solid - settings.young_void
    )
    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    values = (modulus[:, None, None] * ke[None, :, :]).ravel()
    stiffness = coo_matrix((values, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return (stiffness + stiffness.T) * 0.5


def finite_element_analysis(
    physical_density: np.ndarray,
    settings: Top88Settings,
    edof: np.ndarray,
    ke: np.ndarray,
    force: np.ndarray,
    free_dofs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve equilibrium and return displacements, element energies, compliance."""

    stiffness = assemble_stiffness(
        edof, ke, physical_density, settings, force.size
    )
    displacement = np.zeros_like(force)
    k_free = stiffness[free_dofs, :][:, free_dofs]
    displacement[free_dofs] = spsolve(k_free, force[free_dofs])
    element_u = displacement[edof]
    element_energy = np.einsum("ij,jk,ik->i", element_u, ke, element_u)
    modulus = settings.young_void + physical_density**settings.penal * (
        settings.young_solid - settings.young_void
    )
    compliance = float(np.dot(modulus, element_energy))
    return displacement, element_energy, compliance


def optimality_criteria_update(
    design_density: np.ndarray,
    sensitivity: np.ndarray,
    volume_sensitivity: np.ndarray,
    settings: Top88Settings,
    filter_operator: csc_matrix,
    filter_sums: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the move-limited OC update and enforce the volume constraint."""

    lower_multiplier = 0.0
    upper_multiplier = 1.0e9
    physical_density = design_density.copy()
    while (upper_multiplier - lower_multiplier) / (
        upper_multiplier + lower_multiplier
    ) > 1.0e-3:
        multiplier = 0.5 * (lower_multiplier + upper_multiplier)
        ratio = np.maximum(-sensitivity / volume_sensitivity / multiplier, 1.0e-30)
        trial = design_density * np.sqrt(ratio)
        updated = np.maximum(
            0.0,
            np.maximum(
                design_density - settings.move_limit,
                np.minimum(1.0, np.minimum(design_density + settings.move_limit, trial)),
            ),
        )
        if settings.filter_type == 1:
            physical_density = updated
        else:
            physical_density = filter_operator.dot(updated) / filter_sums
        if physical_density.sum() > settings.volfrac * design_density.size:
            lower_multiplier = multiplier
        else:
            upper_multiplier = multiplier
    return updated, physical_density


def validate_settings(settings: Top88Settings) -> None:
    """Reject invalid command-line inputs before numerical work starts."""

    if settings.nelx < 2 or settings.nely < 2:
        raise ValueError("nelx and nely must both be at least 2")
    if not 0.0 < settings.volfrac <= 1.0:
        raise ValueError("volfrac must lie in (0, 1]")
    if settings.penal <= 1.0:
        raise ValueError("penal must be greater than 1")
    if settings.rmin <= 1.0:
        raise ValueError("rmin must be greater than 1")
    if settings.filter_type not in (1, 2):
        raise ValueError("filter_type must be 1 (sensitivity) or 2 (density)")


def topology_optimization(
    settings: Top88Settings, *, verbose: bool = True
) -> Top88Result:
    """Run the SIMP--OC loop until the maximum density change is below tolerance."""

    validate_settings(settings)
    start = time.perf_counter()
    number_of_elements = settings.nelx * settings.nely
    ke = element_stiffness(settings.poisson)
    edof = element_dofs(settings.nelx, settings.nely)
    filter_operator, filter_sums = filter_matrix(
        settings.nelx, settings.nely, settings.rmin
    )
    force, free_dofs = half_mbb_load_and_supports(settings.nelx, settings.nely)

    design_density = np.full(number_of_elements, settings.volfrac)
    physical_density = design_density.copy()
    history: list[dict[str, float | int]] = []

    if verbose:
        print(" iter | compliance | volume | max change | gray fraction")
    for iteration in range(1, settings.max_iterations + 1):
        _, element_energy, compliance = finite_element_analysis(
            physical_density, settings, edof, ke, force, free_dofs
        )
        sensitivity = (
            -settings.penal
            * (settings.young_solid - settings.young_void)
            * physical_density ** (settings.penal - 1.0)
            * element_energy
        )
        volume_sensitivity = np.ones(number_of_elements)

        if settings.filter_type == 1:
            sensitivity = filter_operator.dot(design_density * sensitivity)
            sensitivity /= filter_sums * np.maximum(1.0e-3, design_density)
        else:
            sensitivity = filter_operator.dot(sensitivity / filter_sums)
            volume_sensitivity = filter_operator.dot(
                volume_sensitivity / filter_sums
            )

        updated, updated_physical = optimality_criteria_update(
            design_density,
            sensitivity,
            volume_sensitivity,
            settings,
            filter_operator,
            filter_sums,
        )
        change = float(np.max(np.abs(updated - design_density)))
        design_density = updated
        physical_density = updated_physical
        gray_fraction = float(
            np.mean((physical_density > 0.1) & (physical_density < 0.9))
        )
        record: dict[str, float | int] = {
            "iteration": iteration,
            "compliance": compliance,
            "volume_fraction": float(physical_density.mean()),
            "max_density_change": change,
            "gray_fraction": gray_fraction,
        }
        history.append(record)
        if verbose:
            print(
                f"{iteration:5d} | {compliance:10.4f} |"
                f" {physical_density.mean():6.4f} | {change:10.4f} |"
                f" {gray_fraction:8.4f}"
            )
        if change <= settings.tolerance:
            break
    else:
        raise RuntimeError(
            f"Optimization did not converge within {settings.max_iterations} iterations"
        )

    # Re-evaluate the converged topology so the reported objective matches the
    # final physical density rather than the pre-update design from the last loop.
    _, _, final_compliance = finite_element_analysis(
        physical_density, settings, edof, ke, force, free_dofs
    )
    history[-1]["compliance"] = final_compliance
    density_image = physical_density.reshape((settings.nelx, settings.nely)).T
    return Top88Result(
        density=density_image,
        history=history,
        runtime_seconds=time.perf_counter() - start,
    )


def save_problem_setup(settings: Top88Settings, path: Path) -> None:
    """Draw the half MBB domain, unit load, symmetry edge, and roller support."""

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            settings.nelx,
            settings.nely,
            facecolor="#EAF1F8",
            edgecolor="#173A5E",
            linewidth=2,
        )
    )
    ax.annotate(
        "Normalized unit load",
        xy=(0, settings.nely),
        xytext=(0, settings.nely + 12),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": "#E67826", "lw": 2.5},
        color="#7A3A0B",
        fontsize=10,
    )
    y_locations = np.linspace(4, settings.nely - 4, 7)
    ax.scatter(
        np.full_like(y_locations, -1.5),
        y_locations,
        marker=">",
        s=55,
        color="#1F8A8A",
        label="Horizontal symmetry constraint",
    )
    ax.scatter(
        [settings.nelx],
        [-1.5],
        marker="^",
        s=110,
        color="#1F8A8A",
        label="Vertical roller support",
    )
    ax.text(
        settings.nelx / 2,
        settings.nely / 2,
        f"Half MBB design domain\n{settings.nelx} x {settings.nely} elements",
        ha="center",
        va="center",
        fontsize=13,
        color="#173A5E",
        weight="bold",
    )
    ax.set_xlim(-8, settings.nelx + 5)
    ax.set_ylim(-7, settings.nely + 18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    result: Top88Result, settings: Top88Settings, project_dir: Path
) -> None:
    """Write deterministic figures, tabular history, density, and summary metadata."""

    figure_dir = project_dir / "figures"
    result_dir = project_dir / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    save_problem_setup(settings, figure_dir / "problem_setup.png")

    fig, ax = plt.subplots(figsize=(11, 4.1))
    image = ax.imshow(
        result.density,
        cmap="gray_r",
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        aspect="equal",
    )
    ax.set_title("Optimized half MBB topology (black = solid)")
    ax.set_xlabel("Element column")
    ax.set_ylabel("Element row")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Physical density")
    fig.tight_layout()
    fig.savefig(figure_dir / "final_topology.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    iterations = np.array([row["iteration"] for row in result.history], dtype=int)
    compliance = np.array([row["compliance"] for row in result.history])
    volume = np.array([row["volume_fraction"] for row in result.history])
    change = np.array([row["max_density_change"] for row in result.history])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].plot(iterations, compliance, color="#173A5E", linewidth=2)
    axes[0].set(xlabel="Iteration", ylabel="Normalized compliance", title="Objective convergence")
    axes[0].grid(alpha=0.25)
    axes[1].plot(iterations, volume, color="#1F8A8A", linewidth=2, label="Volume fraction")
    axes[1].axhline(settings.volfrac, color="#E67826", linestyle="--", label="Constraint")
    axes[1].plot(iterations, change, color="#7A3A0B", linewidth=1.5, label="Maximum change")
    axes[1].set(xlabel="Iteration", title="Constraint and stopping metric")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "convergence.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    with (result_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.history[0].keys()))
        writer.writeheader()
        writer.writerows(result.history)
    np.savetxt(result_dir / "final_density.csv", result.density, delimiter=",", fmt="%.8f")

    final = result.history[-1]
    summary = {
        "algorithm": "Python reproduction of the classic 88-line SIMP-OC method",
        "benchmark": "symmetric half MBB beam",
        "settings": asdict(settings),
        "number_of_elements": settings.nelx * settings.nely,
        "number_of_dofs": 2 * (settings.nelx + 1) * (settings.nely + 1),
        "iterations": int(final["iteration"]),
        "final_compliance": float(final["compliance"]),
        "final_volume_fraction": float(final["volume_fraction"]),
        "final_max_density_change": float(final["max_density_change"]),
        "final_gray_fraction": float(final["gray_fraction"]),
        "runtime_seconds": result.runtime_seconds,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (result_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nelx", type=int, default=120)
    parser.add_argument("--nely", type=int, default=40)
    parser.add_argument("--volfrac", type=float, default=0.5)
    parser.add_argument("--penal", type=float, default=3.0)
    parser.add_argument("--rmin", type=float, default=3.5)
    parser.add_argument("--filter-type", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing figures/ and results/ (default: Project 1 root)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Top88Settings(
        nelx=args.nelx,
        nely=args.nely,
        volfrac=args.volfrac,
        penal=args.penal,
        rmin=args.rmin,
        filter_type=args.filter_type,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    result = topology_optimization(settings, verbose=not args.quiet)
    save_outputs(result, settings, args.project_dir)
    final = result.history[-1]
    print(
        "Completed: "
        f"iterations={final['iteration']}, "
        f"compliance={final['compliance']:.6f}, "
        f"volume={final['volume_fraction']:.6f}, "
        f"change={final['max_density_change']:.6f}, "
        f"runtime={result.runtime_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
