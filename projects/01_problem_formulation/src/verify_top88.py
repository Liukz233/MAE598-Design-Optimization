"""Independent numerical checks for the submitted half-MBB benchmark.

Run from the repository root after top88.py. Writes results/verification.json.
These checks assess mechanics and bookkeeping, not optimality or mesh convergence.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from top88 import (
    Top88Settings, assemble_stiffness, element_dofs, element_stiffness,
    finite_element_analysis, half_mbb_load_and_supports,
)


def quadrature_stiffness(nu):
    """Independent 2x2 Gauss integration on a unit square in BL,BR,TR,TL order."""
    d = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]) / (1 - nu**2)
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    ke = np.zeros((8, 8))
    for xi in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
        for eta in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            dn = np.array([
                [-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)],
                [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi],
            ]) / 4
            jac = dn @ xy
            grad = np.linalg.solve(jac, dn)
            b = np.zeros((3, 8))
            b[0, 0::2], b[1, 1::2] = grad[0], grad[1]
            b[2, 0::2], b[2, 1::2] = grad[1], grad[0]
            ke += b.T @ d @ b * np.linalg.det(jac)
    return ke


def main():
    project = Path(__file__).resolve().parents[1]
    summary = json.loads((project / 'results/summary.json').read_text())
    s = Top88Settings(**summary['settings'])
    rho = np.loadtxt(project / 'results/final_density.csv', delimiter=',').T.ravel()
    hist = np.genfromtxt(project / 'results/history.csv', delimiter=',', names=True)
    ke = element_stiffness(s.poisson)
    ke_error = float(np.max(np.abs(ke - quadrature_stiffness(s.poisson))))
    assert ke_error < 1e-12
    edof = element_dofs(s.nelx, s.nely)
    force, free = half_mbb_load_and_supports(s.nelx, s.nely)
    u, _, compliance = finite_element_analysis(rho, s, edof, ke, force, free)
    k = assemble_stiffness(edof, ke, rho, s, force.size)
    reactions = k @ u - force
    residual = float(np.linalg.norm(reactions[free]) / np.linalg.norm(force[free]))
    energy_error = float(abs(compliance - force @ u) / compliance)
    fixed = np.setdiff1d(np.arange(force.size), free)
    boundary_error = float(np.max(np.abs(u[fixed])))
    # Independent force/moment balance with y measured upward from the bottom.
    nodes = np.arange(force.size // 2)
    x = nodes // (s.nely + 1)
    y = s.nely - nodes % (s.nely + 1)
    applied_and_reaction = force.copy()
    applied_and_reaction[fixed] += reactions[fixed]
    balance = applied_and_reaction.reshape(-1, 2).sum(axis=0)
    moment = float(x @ applied_and_reaction[1::2] - y @ applied_and_reaction[0::2])
    assert residual < 1e-7 and energy_error < 1e-7 and boundary_error == 0
    assert np.max(np.abs(balance)) < 1e-7 and abs(moment) < 1e-5
    assert abs(compliance - summary['final_compliance']) < 1e-6
    assert rho.min() >= 0 and rho.max() <= 1 and rho.mean() <= s.volfrac
    assert len(hist) == summary['iterations'] + 1
    assert hist['iteration'][0] == 0 and np.isnan(hist['max_density_change'][0])
    assert np.all(hist['max_density_change'][1:-1] > s.tolerance)
    assert hist['max_density_change'][-1] <= s.tolerance
    assert np.nanmax(hist['max_density_change']) <= s.move_limit + 1e-12

    # Compare the analytical UNFILTERED compliance derivative to finite differences.
    small = replace(s, nelx=6, nely=2)
    se = element_dofs(6, 2)
    sf, sfree = half_mbb_load_and_supports(6, 2)
    srho = np.linspace(0.35, 0.65, 12)
    _, energy, _ = finite_element_analysis(srho, small, se, ke, sf, sfree)
    analytical = -small.penal * (small.young_solid - small.young_void) * srho**(small.penal - 1) * energy
    fd = np.empty(12)
    for e in range(12):
        delta = np.zeros(12)
        delta[e] = 1e-5
        cp = finite_element_analysis(srho + delta, small, se, ke, sf, sfree)[2]
        cm = finite_element_analysis(srho - delta, small, se, ke, sf, sfree)[2]
        fd[e] = (cp - cm) / (2e-5)
    gradient_error = float(np.linalg.norm(fd - analytical) / np.linalg.norm(analytical))
    assert gradient_error < 1e-5

    # A numerical Jensen counterexample on the actual 120x40 design space.
    ex, ey = np.meshgrid(np.arange(s.nelx), np.arange(s.nely), indexing='ij')
    checker = (1 - 2 * ((ex + ey) % 2)).ravel()
    a = 0.5 + 0.1 * checker
    b = 0.5 - 0.1 * checker
    c_a, c_b, c_mid = [finite_element_analysis(z, s, edof, ke, force, free)[2] for z in (a, b, (a+b)/2)]
    gap = c_mid - (c_a + c_b) / 2
    assert gap > 1e-3
    out = {
        'all_checks_passed': True,
        'element_stiffness_max_abs_error_vs_gauss_quadrature': ke_error,
        'free_equilibrium_relative_residual': residual,
        'compliance_work_energy_relative_error': energy_error,
        'prescribed_displacement_max_abs_error': boundary_error,
        'net_force_xy': balance.tolist(),
        'net_moment': moment,
        'right_roller_vertical_reaction': float(reactions[-1]),
        'final_density_min_max': [float(rho.min()), float(rho.max())],
        'final_compliance_recomputed': compliance,
        'history_volume_fraction_min_max': [float(hist['volume_fraction'].min()), float(hist['volume_fraction'].max())],
        'unfiltered_sensitivity_relative_fd_error': gradient_error,
        'nonconvexity_witness': {
            'definition': 'rho_A(ex,ey)=0.5+0.1*(-1)^(ex+ey); rho_B=1-rho_A, with zero-based indices',
            'volume_A': float(a.mean()), 'volume_B': float(b.mean()),
            'compliance_A': c_a, 'compliance_B': c_b,
            'compliance_midpoint': c_mid, 'average_endpoint_compliance': (c_a+c_b)/2,
            'Jensen_violation': gap,
        },
        'scope': 'Checks mechanics, raw sensitivities, saved results, and stopping criterion. No KKT, global-optimality, or mesh-convergence certificate.',
    }
    (project / 'results/verification.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
