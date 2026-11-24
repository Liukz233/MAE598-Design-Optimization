"""Verification ladder for the beam solver and its adjoint gradients.

Each test returns rows of (id, test, quantity, computed, exact, relative error).
The exact values are closed-form results written out in Section 3 of the report,
or values published in the Project 1 report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from beam import Beam, half_mbb_load_and_supports
from problem import CONFIG, BeamDesign

E, NU, FY, N = CONFIG["E"], CONFIG["nu"], CONFIG["f_y"], CONFIG["n"]
P1 = Path(__file__).resolve().parents[2] / "01_problem_formulation" / "results"


def row(vid, test, quantity, computed, exact, tol=1e-2, scale=None):
    """One ladder entry. The error is relative to |exact|, or to `scale` when exact is 0."""
    err = abs(computed - exact) / (abs(exact) if scale is None else scale)
    return {"id": vid, "test": test, "quantity": quantity, "computed": float(computed),
            "exact": float(exact), "rel_error": float(err), "pass": bool(err < tol)}


def edge_traction(b: Beam, traction) -> np.ndarray:
    """Nodal forces for a horizontal traction (N/m, function of y) on the right edge."""
    F = np.zeros(b.ndof)
    nodes = np.flatnonzero(np.abs(b.xy[:, 0] - b.nelx * b.a) < 1e-12)
    nodes = nodes[np.argsort(b.xy[nodes, 1])]
    p = traction(b.xy[nodes, 1])
    for j in range(nodes.size - 1):            # exact for a traction that is linear in y
        F[2 * nodes[j]] += b.a * (2 * p[j] + p[j + 1]) / 6
        F[2 * nodes[j + 1]] += b.a * (p[j] + 2 * p[j + 1]) / 6
    return F


def v1_patch():
    """Uniform tension: bilinear elements must reproduce it exactly, for both material laws."""
    L, H, t, s = 0.30, 0.10, 0.008, 0.9 * FY
    rows = []
    for nonlinear, label in ((False, "linear material"), (True, "Ramberg-Osgood material")):
        b = Beam(12, 4, L / 12, E, NU, FY, N, nonlinear)
        u, _ = b.solve(np.full(b.ne, t), edge_traction(b, lambda y: np.full_like(y, s * t)))
        strain = s / E + (0.002 * (s / FY) ** N if nonlinear else 0.0)
        right_top = 2 * (b.nely + 1) * b.nelx                 # x displacement, top-right node
        rows.append(row("V5" if nonlinear else "V1", f"Uniform tension, {label}",
                        "elongation (m)", u[right_top], strain * L, tol=1e-8))
        if not nonlinear:
            rows.append(row("V1", f"Uniform tension, {label}", "lateral contraction (m)",
                            u[right_top + 1] - u[right_top + 1 + 2 * b.nely], -NU * strain * H, tol=1e-8))
    return rows


def v2_bending():
    """Pure bending, an exact solution of plane-stress elasticity: curvature M/(EI)."""
    L, H, t, M = 0.60, 0.20, 0.008, 2.0e3
    inertia = t * H**3 / 12
    exact = M * L**2 / (2 * E * inertia)                      # rise of the mid-depth line over L
    sizes, errors, last = [], [], None
    for nely in (2, 4, 8, 16, 32):
        b = Beam(3 * nely, nely, H / nely, E, NU, nonlinear=False)
        # sigma_xx = -M (y - H/2) / I, applied as a line traction sigma_xx * t
        u, _ = b.solve(np.full(b.ne, t), edge_traction(b, lambda y: -M * (y - H / 2) / inertia * t))
        mid_left, mid_right = nely // 2, (nely + 1) * b.nelx + nely // 2
        last = abs(u[2 * mid_right + 1] - u[2 * mid_left + 1])
        sizes.append(b.a)
        errors.append(abs(last - exact) / exact)
    slope = np.polyfit(np.log(sizes[1:]), np.log(errors[1:]), 1)[0]
    return [row("V1", "Pure bending, finest mesh", "deflection of the mid-depth line (m)", last, exact, tol=1e-2),
            row("V2", "Pure bending, mesh refinement", "convergence rate of the deflection error", slope, 2.0,
                tol=0.05)], {"h_mm": [1e3 * s for s in sizes], "error": errors, "slope": float(slope)}


def v3_reaction():
    """With a real roller instead of the support pad, the roller must carry half the load."""
    rows = []
    for nonlinear, label in ((False, "linear"), (True, "nonlinear")):
        d = BeamDesign(nonlinear=nonlinear)
        b = d.beam
        F = d.F.copy()
        F[1::2] = np.minimum(F[1::2], 0.0)                    # keep the load pad, drop the support pad
        t_el = d.thickness(np.random.default_rng(3).uniform(0, 1, d.n))[d.tile_of_element]
        u, _ = b.solve(t_el, F)
        rows.append(row("V3", f"Global force balance, {label} material", "reaction at the roller (N)",
                        b.internal(u, t_el)[b.ndof - 1], 0.5 * CONFIG["P"], tol=1e-8))
    return rows


def v4_invariance():
    """Rigid-body modes, mirror symmetry, and agreement with Project 1."""
    b = Beam(6, 2, 0.01, E, NU, nonlinear=False)
    lam = np.linalg.eigvalsh(b.tangent(np.zeros(b.ndof), np.full(b.ne, 0.005)).toarray())
    rows = [row("V4", "Unsupported stiffness matrix", "number of zero eigenvalues (rigid-body modes)",
                int(np.sum(lam < 1e-9 * lam[-1])), 3, tol=1e-12)]

    # mirror the beam about mid-depth: thickness and loads flip, the stress field must flip with them
    d = BeamDesign()
    b = d.beam
    x = np.random.default_rng(4).uniform(0, 1, d.n)
    ref = d.responses(d.solve(x)[0])["utilization"].reshape(d.tx, d.ty)
    node = np.arange(b.ndof // 2)
    mirrored = (node // (b.nely + 1)) * (b.nely + 1) + (b.nely - node % (b.nely + 1))
    F = np.zeros(b.ndof)
    F[2 * mirrored + 1] = -d.F[1::2]
    t_tile = d.thickness(x).reshape(d.tx, d.ty)[:, ::-1].ravel()
    u, _ = b.solve(t_tile[d.tile_of_element], F)
    flipped = d.responses(u)["utilization"].reshape(d.tx, d.ty)[:, ::-1]
    rows.append(row("V4", "Mirror the beam about mid-depth", "largest change in stress utilization",
                    np.abs(flipped - ref).max(), 0.0, tol=1e-9, scale=ref.max()))

    # Project 1: same mesh, supports and unit load; modulus field of the final SIMP design
    rho = np.loadtxt(P1 / "final_density.csv", delimiter=",").T.ravel()
    published = json.loads((P1 / "summary.json").read_text())["final_compliance"]
    b = Beam(120, 40, 1.0, 1.0, 0.3, nonlinear=False)
    F = half_mbb_load_and_supports(120, 40)[0]
    u, _ = b.solve(1e-9 + rho**3 * (1 - 1e-9), F)
    rows.append(row("V4", "Project 1 final design, Project 1 load", "compliance (Project 1 units)",
                    F @ u, published, tol=1e-8))
    return rows


def v5_nonlinear():
    """Newton convergence order and the energy form of the state equation."""
    d = BeamDesign()
    b = d.beam
    x = np.full(d.n, 0.2)
    _, info = d.solve(x)
    r = np.array(info["history"])
    q = r[r > 1e-10][-3:]                                     # the last step ends at round-off
    order = np.log(q[2] / q[1]) / np.log(q[1] / q[0])
    rows = [row("V5", "Newton's method on the design model", "observed convergence order", order, 2.0, tol=0.2)]

    rng = np.random.default_rng(5)
    t_el = d.thickness(rng.uniform(0, 1, d.n))[d.tile_of_element]
    u = 2e-3 * rng.normal(size=b.ndof)
    v = rng.normal(size=b.ndof)
    step = 1e-7
    dPi = (b.energy(u + step * v, t_el, d.F) - b.energy(u - step * v, t_el, d.F)) / (2 * step)
    rows.append(row("V5", "Residual as the gradient of the potential energy", "directional derivative (J)",
                    (b.internal(u, t_el) - d.F) @ v, dPi, tol=1e-6))
    return rows, {"residuals": r.tolist(), "order": float(order)}


def v6_gradients():
    """Adjoint directional derivatives against central differences (Taylor test)."""
    rows, sweeps = [], {}
    steps = 10.0 ** -np.arange(1, 9)
    rng = np.random.default_rng(6)
    for nonlinear in (True, False):
        d = BeamDesign(nonlinear=nonlinear)
        x0 = rng.uniform(0.2, 0.8, d.n)
        base = d.evaluate(x0)
        label = "nonlinear material" if nonlinear else "linear material"
        for name in d.names:
            worst = (-1.0, None, None)  # (error, adjoint, finite difference)
            for trial in range(3):
                v = rng.normal(size=d.n)
                ad = base["dg_" + name] @ v
                fd = []
                for s in steps:
                    d.u_last = None
                    gp = d.evaluate(x0 + s * v, gradients=False)["g_" + name]
                    d.u_last = None
                    gm = d.evaluate(x0 - s * v, gradients=False)["g_" + name]
                    fd.append((gp - gm) / (2 * s))
                errs = np.abs(np.array(fd) - ad) / abs(ad)
                if trial == 0 and nonlinear:
                    sweeps[name] = errs.tolist()
                i = int(np.argmin(errs))
                if errs[i] > worst[0]:
                    worst = (errs[i], ad, fd[i])
            rows.append(row("V6", f"Adjoint vs central difference, {label}",
                            f"directional derivative of g_{name} (worst of 3 directions)",
                            worst[1], worst[2], tol=1e-5))

    # self-adjoint special case (Project 1): for the compliance F.u of the linear model the
    # adjoint is -u, so dC/dt_e = -u_e^T k_e u_e without an adjoint solve
    d = BeamDesign(nonlinear=False)
    b = d.beam
    u, info = d.solve(rng.uniform(0, 1, d.n))
    general = b.adjoint_gradient(u, info["lu"], d.F)
    closed = -np.einsum("ei,ei->e", u[b.edof], b.element_forces(u)[0])
    rows.append(row("V6", "Self-adjoint case, Project 1 formula", "largest difference in the compliance sensitivity",
                    np.abs(general - closed).max(), 0.0, tol=1e-9, scale=np.abs(closed).max()))
    slope = np.polyfit(np.log(steps[:3]), np.log(sweeps["deflection"][:3]), 1)[0]
    return rows, {"steps": steps.tolist(), "errors": sweeps, "slope": float(slope)}


def run_ladder():
    patch = v1_patch()
    bending, convergence = v2_bending()
    r5, newton = v5_nonlinear()
    r6, taylor = v6_gradients()
    rows = ([r for r in patch if r["id"] == "V1"] + bending + v3_reaction() + v4_invariance()
            + [r for r in patch if r["id"] == "V5"] + r5 + r6)
    return rows, {"convergence": convergence, "newton": newton, "taylor": taylor}


if __name__ == "__main__":
    rows, extra = run_ladder()
    for r in rows:
        print(f"{r['id']} {'ok  ' if r['pass'] else 'FAIL'} {r['test']:<50s} {r['quantity']:<60s} "
              f"{r['computed']:.9g}  {r['exact']:.9g}  {r['rel_error']:.2e}")
    print(extra["convergence"], extra["newton"], extra["taylor"]["slope"])
