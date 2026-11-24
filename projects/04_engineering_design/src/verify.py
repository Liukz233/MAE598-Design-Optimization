"""Verification ladder for the mechanism solver and its adjoint gradients.

Each test returns rows of (id, test, quantity, computed, exact, relative error).
The exact values are closed-form results written out in Section 3 of the report,
an independent integration of the elastica, or values published in Project 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from mech import Mechanism
from problem import CONFIG, Gripper

P1 = Path(__file__).resolve().parents[2] / "01_problem_formulation"


def row(vid, test, quantity, computed, exact, tol=1e-2, scale=None):
    """One ladder entry. The error is relative to |exact|, or to `scale` when exact is 0."""
    err = abs(computed - exact) / (abs(exact) if scale is None else scale)
    return {"id": vid, "test": test, "quantity": quantity, "computed": float(computed),
            "exact": float(exact), "rel_error": float(err), "pass": bool(err < tol)}


def plate(nelx, nely, L, nonlinear, nu=0.3, clamp=False):
    """A solid plate of unit modulus and thickness; left edge on rollers (or clamped)."""
    m = Mechanism(nelx, nely, L / nelx, 1.0, 1.0, nu, nonlinear=nonlinear)
    left = np.flatnonzero(m.xy[:, 0] < 1e-12)
    corner = left[np.argmin(m.xy[left, 1])]
    m.fixed = np.concatenate([2 * left, 2 * left + 1]) if clamp else np.concatenate([2 * left, [2 * corner + 1]])
    right = np.flatnonzero(np.abs(m.xy[:, 0] - L) < 1e-12)
    return m, right[np.argsort(m.xy[right, 1])]


def edge_load(m, nodes, traction, direction=0):
    """Nodal forces for a traction (force per unit area, function of y) on the right edge."""
    F = np.zeros(m.ndof)
    p = traction(m.xy[nodes, 1]) * m.t
    for j in range(nodes.size - 1):                # exact for a traction that is linear in y
        F[2 * nodes[j] + direction] += m.a * (2 * p[j] + p[j + 1]) / 6
        F[2 * nodes[j + 1] + direction] += m.a * (p[j] + 2 * p[j + 1]) / 6
    return F


def v1_linear():
    """Uniform tension (exact for bilinear elements) and pure bending (exact solution of elasticity)."""
    L, H, s, nu = 3.0, 1.0, 1e-3, 0.3
    m, right = plate(12, 4, L, nonlinear=False)
    u, _ = m.solve(np.ones(m.ne), edge_load(m, right, lambda y: np.full_like(y, s)))
    rows = [row("V1", "Uniform tension, small strain", "elongation", u[2 * right[0]], s * L, tol=1e-8),
            row("V1", "Uniform tension, small strain", "lateral contraction", u[2 * right[-1] + 1] - u[2 * right[0] + 1],
                -nu * s * H, tol=1e-8)]
    M, inertia = 1e-3, H**3 / 12
    exact = M * L**2 / (2 * inertia)                # rise of the mid-depth line over the length
    sizes, errors, last = [], [], None
    for nely in (2, 4, 8, 16, 32):
        m, right = plate(3 * nely, nely, L, nonlinear=False)
        u, _ = m.solve(np.ones(m.ne), edge_load(m, right, lambda y: -M * (y - H / 2) / inertia))
        mid_left, mid_right = nely // 2, (nely + 1) * m.nelx + nely // 2
        last = abs(u[2 * mid_right + 1] - u[2 * mid_left + 1])
        sizes.append(m.a)
        errors.append(abs(last - exact) / exact)
    slope = np.polyfit(np.log(sizes[1:]), np.log(errors[1:]), 1)[0]
    rows += [row("V1", "Pure bending, finest mesh", "deflection of the mid-depth line", last, exact),
             row("V2", "Pure bending, mesh refinement", "convergence rate of the deflection error", slope, 2.0,
                 tol=0.05)]
    return rows, {"h": sizes, "error": errors, "slope": float(slope)}


def v3_balance(x):
    """The clamp must carry the whole actuator force."""
    rows = []
    for nonlinear, label in ((False, "linear"), (True, "large-displacement")):
        g = Gripper(nonlinear=nonlinear)
        m = g.m
        rho = g.density(x)
        u, _ = g.solve(rho)
        R = m.internal(u, rho)
        clamp_x = [d for d in m.fixed if d % 2 == 0]
        rows.append(row("V3", f"Global force balance, {label} model", "horizontal reaction at the clamp (N)",
                        -R[clamp_x].sum(), 0.5 * CONFIG["F_in"], tol=1e-7))
    return rows


def v4_invariance():
    """Rigid-body modes, finite rotation, and agreement with Project 1."""
    m = Mechanism(6, 3, 0.5, 1.0, 1.0, 0.3)
    lam = np.linalg.eigvalsh(m.tangent(np.zeros(m.ndof), np.ones(m.ne)).toarray())
    rows = [row("V4", "Unsupported stiffness matrix", "number of zero eigenvalues (rigid-body modes)",
                int(np.sum(lam < 1e-9 * lam[-1])), 3, tol=1e-12)]
    theta = np.deg2rad(40.0)
    Rm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    u = (m.xy @ Rm.T - m.xy).ravel()
    lin = Mechanism(6, 3, 0.5, 1.0, 1.0, 0.3, nonlinear=False)
    rows.append(row("V4", "Rigid rotation by 40 degrees", "largest internal force",
                    np.abs(m.internal(u, np.ones(m.ne))).max(), 0.0, tol=1e-10,
                    scale=np.abs(lin.internal(u, np.ones(m.ne))).max()))

    # Project 1: same mesh, supports and unit load; modulus field of its final SIMP design
    import top88
    rho = np.loadtxt(P1 / "results" / "final_density.csv", delimiter=",").T.ravel()
    published = json.loads((P1 / "results" / "summary.json").read_text())["final_compliance"]
    m = Mechanism(120, 40, 1.0, 1.0, 1.0, 0.3, E_min=1e-9, nonlinear=False)
    F, free = top88.half_mbb_load_and_supports(120, 40)
    m.fixed = np.setdiff1d(np.arange(m.ndof), free)
    u, _ = m.solve(rho, F)
    rows.append(row("V4", "Project 1 final design, Project 1 load", "compliance (Project 1 units)",
                    F @ u, published, tol=1e-8))
    return rows


def elastica_tip(alpha):
    """Tip deflection w/L and shortening u/L of a cantilever with end load P = alpha EI / L^2,
    from the elastica equation theta'' = -alpha cos(theta), theta(0) = 0, theta'(1) = 0."""
    def shoot(curvature0):
        sol = solve_ivp(lambda s, y: [y[1], -alpha * np.cos(y[0]), np.sin(y[0]), np.cos(y[0])], [0, 1],
                        [0.0, curvature0, 0.0, 0.0], rtol=1e-11, atol=1e-13)
        return sol.y[:, -1]
    k0 = brentq(lambda k: shoot(k)[1], 0.0, alpha)
    _, _, w, x = shoot(k0)
    return w, 1 - x


def v5_nonlinear(x):
    """Finite-strain benchmarks, Newton convergence order, and the energy form."""
    rows = []
    # (a) uniform tension with a dead load P per unit undeformed area: P = lam (lam^2 - 1) / 2 for E = 1
    L, P = 3.0, 0.2
    m, right = plate(12, 4, L, nonlinear=True)
    u, _ = m.solve(np.ones(m.ne), edge_load(m, right, lambda y: np.full_like(y, P)), steps=4)
    lam = [r.real for r in np.roots([0.5, 0.0, -0.5, -P]) if abs(r.imag) < 1e-12 and r.real > 0][0]
    rows.append(row("V5", "Uniform tension at 15% stretch", "elongation", u[2 * right[0]], (lam - 1) * L, tol=1e-7))

    # (b) cantilever under an end load, against the elastica
    L, H, alpha, nely = 1.0, 0.05, 2.0, 8
    m, right = plate(20 * nely, nely, L, nonlinear=True, clamp=True)
    load = alpha * (H**3 / 12) / L**2                          # E = 1, unit thickness
    u, _ = m.solve(np.ones(m.ne), edge_load(m, right, lambda y: np.full_like(y, -load / H), direction=1), steps=8)
    mid = right[nely // 2]
    w, shortening = elastica_tip(alpha)
    rows += [row("V5", "Cantilever, end load 2 EI/L^2", "tip deflection / length", -u[2 * mid + 1] / L, w, tol=1e-2),
             row("V5", "Cantilever, end load 2 EI/L^2", "tip shortening / length", -u[2 * mid] / L, shortening,
                 tol=2e-2)]

    # (c) the tangent matrix is the derivative of the residual, and Newton's method on the gripper
    g = Gripper()
    m = g.m
    rho = g.density(x)
    g.u_last = None
    u, info = g.solve(rho)
    from_rest = info["history"]
    u90, _ = m.solve(rho, 0.9 * g.F, u0=u)
    r = np.array(m.solve(rho, g.F, u0=u90)[1]["history"])
    rng = np.random.default_rng(4)
    v = rng.normal(size=m.ndof)
    ut = u + 0.1 * np.abs(u).max() * rng.normal(size=m.ndof)
    step = 1e-6 * np.abs(u).max()
    dR = (m.internal(ut + step * v, rho) - m.internal(ut - step * v, rho)) / (2 * step)
    Kv = m.tangent(ut, rho) @ v
    rows.append(row("V5", "Tangent matrix against a central difference of the residual",
                    "largest difference in K_T v", np.abs(Kv - dR).max(), 0.0, tol=1e-6, scale=np.abs(dR).max()))

    # (d) the residual is the gradient of the potential energy
    rng = np.random.default_rng(5)
    v = rng.normal(size=m.ndof)
    v[m.fixed] = 0.0
    step = 1e-6 * np.abs(u).max()
    dPi = (m.energy(u + step * v, rho, g.F) - m.energy(u - step * v, rho, g.F)) / (2 * step)
    ut = u + 0.1 * np.abs(u).max() * rng.normal(size=m.ndof)
    dPi = (m.energy(ut + step * v, rho, g.F) - m.energy(ut - step * v, rho, g.F)) / (2 * step)
    rows.append(row("V5", "Residual as the gradient of the potential energy", "directional derivative (J/m)",
                    (m.internal(ut, rho) - g.F) @ v, dPi, tol=1e-6))
    return rows, {"residuals": r.tolist(), "from_rest": list(map(float, from_rest))}


def v6_gradients():
    """Adjoint directional derivatives against central differences (Taylor test)."""
    rows, sweeps = [], {}
    steps = 10.0 ** -np.arange(1, 8)
    rng = np.random.default_rng(6)
    names = {"f": "objective", "g_stroke": "stroke constraint", "g_stress": "stress constraint"}
    for nonlinear in (True, False):
        g = Gripper(nonlinear=nonlinear)
        x0 = rng.uniform(0.3, 0.9, g.n)
        base = g.evaluate(x0)
        label = "large displacement" if nonlinear else "linear"
        for key, name in names.items():
            dkey = "df" if key == "f" else "d" + key
            worst = (-1.0, None, None)
            for trial in range(2):
                v = rng.normal(size=g.n)
                ad = base[dkey] @ v
                fd = []
                for s in steps:
                    g.u_last = None
                    gp = g.evaluate(x0 + s * v, gradients=False)[key]
                    g.u_last = None
                    gm = g.evaluate(x0 - s * v, gradients=False)[key]
                    fd.append((gp - gm) / (2 * s))
                errs = np.abs(np.array(fd) - ad) / abs(ad)
                if trial == 0 and nonlinear:
                    sweeps[name] = errs.tolist()
                i = int(np.argmin(errs))
                if errs[i] > worst[0]:
                    worst = (errs[i], ad, fd[i])
            rows.append(row("V6", f"Adjoint vs central difference, {label}",
                            f"directional derivative of the {name} (worst of 2 directions)",
                            worst[1], worst[2], tol=1e-5))

    # self-adjoint special case (Project 1): linear model, compliance F.u. The adjoint is -u and
    # dc/drho_e = -p rho^(p-1) (E0 - Emin) u_e^T k0 u_e, the sensitivity used by top88
    g = Gripper(nonlinear=False)
    m = g.m
    rho = g.density(rng.uniform(0.3, 0.9, g.n))
    u, info = g.solve(rho)
    general = m.adjoint_gradient(u, rho, info["lu"], g.F)
    ue = u[m.edof]
    closed = -m.modulus(rho)[1] * np.einsum("ea,ab,eb->e", ue, m.KL, ue)
    rows.append(row("V6", "Self-adjoint case, Project 1 formula", "largest difference in the compliance sensitivity",
                    np.abs(general - closed).max(), 0.0, tol=1e-9, scale=np.abs(closed).max()))
    slope = np.polyfit(np.log(steps[:3]), np.log(sweeps["objective"][:3]), 1)[0]
    return rows, {"steps": steps.tolist(), "errors": sweeps, "slope": float(slope)}


def run_ladder(x=None):
    """x: a design to use for the checks on the gripper model (default: uniform density)."""
    x = np.full(CONFIG["nelx"] * CONFIG["nely"], CONFIG["volume_fraction"]) if x is None else x
    r1, convergence = v1_linear()
    r5, newton = v5_nonlinear(x)
    r6, taylor = v6_gradients()
    rows = r1 + v3_balance(x) + v4_invariance() + r5 + r6
    return rows, {"convergence": convergence, "newton": newton, "taylor": taylor}


if __name__ == "__main__":
    rows, extra = run_ladder()
    for r in rows:
        print(f"{r['id']} {'ok  ' if r['pass'] else 'FAIL'} {r['test']:<50s} {r['quantity']:<62s} "
              f"{r['computed']:.9g}  {r['exact']:.9g}  {r['rel_error']:.2e}")
    print(extra["convergence"]["slope"], extra["newton"], extra["taylor"]["slope"])
