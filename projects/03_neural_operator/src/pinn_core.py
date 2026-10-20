"""Energy-based PINN for the half-MBB equilibrium solve of Projects 1 and 2.

The network maps a node position to its displacement. Training minimizes the
same total potential energy as Project 2,

    Pi(u) = 1/2 u^T K u - F^T u,

evaluated exactly with the Project 1 four-node elements, plus a Dirichlet
penalty on the supported degrees of freedom (as in label-free, energy-trained
TO surrogates). The FE direct solve is the reference.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import spsolve

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

DTYPE = torch.float32  # training precision; metrics are computed in float64
torch.set_default_dtype(DTYPE)


def set_precision(dtype):
    """float32 for Adam runs (2x faster on CPU); float64 for L-BFGS line searches."""
    global DTYPE
    DTYPE = dtype
    torch.set_default_dtype(dtype)


def load_p1_density() -> np.ndarray:
    rho = np.loadtxt(P1 / "results" / "final_density.csv", delimiter=",")
    return rho.ravel(order="F")  # P1 element order: ex * nely + ey


@dataclass
class Problem:
    nx: int
    ny: int
    rho: np.ndarray
    emin: float
    modulus: np.ndarray        # element Young's moduli, SIMP p = 3
    ke: np.ndarray             # 8x8 unit-modulus element matrix
    edof: np.ndarray           # element -> global dof table
    force: np.ndarray          # full load vector
    free: np.ndarray
    fixed: np.ndarray
    xy: np.ndarray             # node coordinates, x right, y up, beam units
    u_ref: np.ndarray          # full reference displacement (fixed dofs = 0)
    compliance_ref: float
    solve_seconds: float
    struct_dofs: np.ndarray    # dofs of nodes touching an element with rho >= 0.5
    K_free: object = None

    @property
    def ndof(self):
        return self.force.size

    @property
    def length(self):
        return 120.0           # physical beam length; element size 120 / nx

    @property
    def height(self):
        return 40.0


def build_problem(nx=120, ny=40, rho=None, emin=1e-9, penal=3.0, solve_repeats=5,
                  load_width=None):
    """load_width=None: the P1 unit point load. load_width=a: the same unit total
    load spread uniformly over the top edge for 0 <= x <= a (consistent nodal
    loads), which removes the point-load singularity."""
    rho = np.full(nx * ny, 0.5) if rho is None else np.asarray(rho, float)
    settings = Top88Settings(nelx=nx, nely=ny, young_void=emin, penal=penal)
    force, free = half_mbb_load_and_supports(nx, ny)
    if load_width is not None:
        h = 120.0 / nx
        n_el = int(round(load_width / h))
        assert abs(n_el * h - load_width) < 1e-9, "load width must be a whole number of elements"
        q = 1.0 / load_width                              # traction, unit total force
        force[:] = 0.0
        for ex in range(n_el):                            # top-edge node of column ex: (ny+1) ex
            for node in ((ny + 1) * ex, (ny + 1) * (ex + 1)):
                force[2 * node + 1] -= 0.5 * q * h
    fixed = np.setdiff1d(np.arange(force.size), free)
    edof = element_dofs(nx, ny)
    ke = element_stiffness(settings.poisson)
    K = assemble_stiffness(edof, ke, rho, settings, force.size)
    Kff = K[free, :][:, free].tocsc()
    times = []
    for _ in range(solve_repeats):
        t0 = time.perf_counter()
        uf = spsolve(Kff, force[free])
        times.append(time.perf_counter() - t0)
    u = np.zeros(force.size)
    u[free] = uf
    modulus = emin + rho**penal * (1.0 - emin)
    h = 120.0 / nx
    ex, ey = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    xy = np.column_stack([ex.ravel() * h, (ny - ey.ravel()) * h])  # node n = (ny+1) ex + ey
    solid = rho >= 0.5
    struct_nodes = np.unique(edof[solid][:, 0::2] // 2)
    struct_dofs = np.sort(np.concatenate([2 * struct_nodes, 2 * struct_nodes + 1]))
    struct_dofs = np.intersect1d(struct_dofs, free)
    return Problem(nx, ny, rho, emin, modulus, ke, edof, force, free, fixed, xy,
                   u, float(force @ u), float(np.median(times)), struct_dofs, Kff)


class TorchEnergy:
    """Exact FE strain energy, load work and Dirichlet penalty in torch."""

    def __init__(self, prob: Problem):
        self.prob = prob
        self.edof = torch.as_tensor(prob.edof)
        self.ke = torch.as_tensor(prob.ke, dtype=DTYPE)
        self.modulus = torch.as_tensor(prob.modulus, dtype=DTYPE)
        self.force = torch.as_tensor(prob.force, dtype=DTYPE)
        self.fixed = torch.as_tensor(prob.fixed)
        self.solid = torch.as_tensor(prob.rho >= 0.5)

    def element_energy(self, u_full):
        ue = u_full[self.edof]                                   # (Ne, 8)
        # Remove each element's rigid translation before the quadratic form. The
        # element matrix annihilates translations exactly, so the value is
        # unchanged in exact arithmetic; in float32 it avoids cancellation
        # between large displacements and small strains (without this, the
        # float32 strain energy at the FE solution is off by 1% on the 120x40
        # mesh and 5% on 240x80).
        ue = torch.stack([ue[:, 0::2] - ue[:, 0::2].mean(1, keepdim=True),
                          ue[:, 1::2] - ue[:, 1::2].mean(1, keepdim=True)], dim=2).reshape(-1, 8)
        return 0.5 * self.modulus * torch.einsum("ei,ij,ej->e", ue, self.ke, ue)

    def terms(self, u_full):
        ee = self.element_energy(u_full)
        strain = ee.sum()
        work = self.force @ u_full
        bc = (u_full[self.fixed] ** 2).sum()
        return strain, work, bc, ee


def beam_scale(prob: Problem) -> float:
    """Euler-Bernoulli midspan deflection of the full simply supported beam.

    Span 2L, midspan load 2F (symmetry), solid modulus, unit thickness:
    delta = (2F)(2L)^3 / (48 E0 I), I = H^3 / 12. About 108 here.
    Used only to make the network output O(1); it uses no FE information.
    """
    inertia = prob.height**3 / 12.0
    return 2.0 * (2.0 * prob.length) ** 3 / (48.0 * inertia)


class FieldNet(torch.nn.Module):
    """MLP (x, y) -> (u_x, u_y), with optional Fourier features and hard BCs."""

    def __init__(self, width=64, depth=4, fourier=0, sigma=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fourier = fourier
        if fourier:
            self.register_buffer("B", sigma * torch.randn(2, fourier, generator=g))
            n_in = 2 * fourier
        else:
            n_in = 2
        layers = []
        for i in range(depth):
            layers += [torch.nn.Linear(n_in if i == 0 else width, width), torch.nn.Tanh()]
        layers.append(torch.nn.Linear(width, 2))
        self.net = torch.nn.Sequential(*layers)
        torch.manual_seed(seed)
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, s):
        if self.fourier:
            z = 2 * math.pi * s @ self.B
            s = torch.cat([torch.sin(z), torch.cos(z)], dim=1)
        return self.net(s)


class Model:
    """Wraps a FieldNet into a full nodal displacement vector."""

    def __init__(self, prob: Problem, net: FieldNet, hard_bc=False, out_scale=None,
                 dof_scale=None):
        self.prob = prob
        self.net = net
        self.hard_bc = hard_bc
        xy = torch.as_tensor(prob.xy, dtype=DTYPE)
        self.s = torch.stack([xy[:, 0] / prob.length, xy[:, 1] / prob.height], 1)
        self.uc = beam_scale(prob) if out_scale is None else out_scale
        self.dof_scale = None if dof_scale is None else torch.as_tensor(dof_scale, dtype=DTYPE)
        if hard_bc:
            # u_x = 0 on x = 0 ; u_y = 0 at (L, 0): multiply by distance functions
            self.gx = self.s[:, 0:1]
            d = torch.sqrt((self.s[:, 0:1] - 1.0) ** 2 + self.s[:, 1:2] ** 2)
            self.gy = torch.clamp(d, max=1.0)

    def parameters(self):
        return self.net.parameters()

    def to(self, dtype):
        """Cast network and fixed tensors, e.g. float32 Adam -> float64 L-BFGS."""
        self.net.to(dtype)
        self.s = self.s.to(dtype)
        if self.hard_bc:
            self.gx, self.gy = self.gx.to(dtype), self.gy.to(dtype)
        if self.dof_scale is not None:
            self.dof_scale = self.dof_scale.to(dtype)
        return self

    def __call__(self):
        out = self.uc * self.net(self.s)
        if self.hard_bc:
            out = torch.cat([out[:, 0:1] * self.gx, out[:, 1:2] * self.gy], 1)
        u = out.reshape(-1)                          # interleaved [ux0, uy0, ux1, ...]
        if self.dof_scale is not None:
            u = u * self.dof_scale
        return u


def full_stiffness(prob: Problem):
    from scipy.sparse import coo_matrix
    rows = np.repeat(prob.edof, 8, axis=1).ravel()
    cols = np.tile(prob.edof, (1, 8)).ravel()
    vals = (prob.modulus[:, None, None] * prob.ke[None]).ravel()
    K = coo_matrix((vals, (rows, cols)), shape=(prob.ndof, prob.ndof)).tocsr()
    return 0.5 * (K + K.T)


def metrics(prob: Problem, u: np.ndarray) -> dict:
    ref = prob.u_ref
    free = prob.free
    sd = prob.struct_dofs
    err = u - ref
    comp = float(prob.force @ u)
    return {
        "rel_err_all": float(np.linalg.norm(err[free]) / np.linalg.norm(ref[free])),
        "rel_err_struct": float(np.linalg.norm(err[sd]) / np.linalg.norm(ref[sd])),
        "rel_compliance_err": float(abs(comp - prob.compliance_ref) / prob.compliance_ref),
        "bc_violation": float(np.abs(u[prob.fixed]).max() / np.abs(ref).max()),
    }


def train(prob: Problem, model: Model, *, mode="energy", steps=20000, lr=1e-3,
          lr_final=None, bc_weight=1.0e3, optimizer="adam", log_every=200,
          lbfgs_iters=None, callback=None):
    """mode='energy': minimize Pi + bc_weight * sum(u_fixed^2).
    mode='supervised': minimize ||u - u_ref||^2 / ||u_ref||^2 (expressivity test).

    bc_weight is a penalty spring stiffness (force/length), E0 = 1.
    """
    E = TorchEnergy(prob)
    u_ref = torch.as_tensor(prob.u_ref, dtype=DTYPE)
    pi_ref = -0.5 * prob.compliance_ref
    hist = []
    params = list(model.parameters())
    sched = None
    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr)
        if lr_final is not None:
            gamma = (lr_final / lr) ** (1.0 / steps)
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50,
                                tolerance_grad=0, tolerance_change=0,
                                line_search_fn="strong_wolfe")
    else:
        raise ValueError(optimizer)

    def loss_fn():
        u = model()
        if mode == "supervised":
            loss = ((u - u_ref) ** 2).sum() / (u_ref ** 2).sum()
            strain, work, bc, _ = E.terms(u)
            return loss, (strain, work, bc, u)
        strain, work, bc, _ = E.terms(u)
        pen = 0.0 if model.hard_bc else bc_weight * bc
        return strain - work + pen, (strain, work, bc, u)

    K64 = full_stiffness(prob)
    t0 = time.perf_counter()
    n_outer = steps if optimizer == "adam" else lbfgs_iters
    for it in range(n_outer + 1):
        if it % log_every == 0 or it == n_outer:
            with torch.no_grad():
                loss, (strain, work, bc, u) = loss_fn()
            un = u.numpy().astype(np.float64)
            strain64 = 0.5 * un @ (K64 @ un)                     # logged in float64
            work64 = float(prob.force @ un)
            row = {"iter": it, "seconds": time.perf_counter() - t0,
                   "loss": float(loss), "strain_energy": strain64,
                   "load_work": work64, "bc_penalty": float(bc_weight * bc),
                   "energy_gap": (strain64 - work64 - pi_ref) / abs(pi_ref)}
            row.update(metrics(prob, un))
            hist.append(row)
            if callback:
                callback(row)
        if it == n_outer:
            break
        if optimizer == "adam":
            opt.zero_grad()
            loss, _ = loss_fn()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        else:
            def closure():
                opt.zero_grad()
                loss, _ = loss_fn()
                loss.backward()
                return loss
            opt.step(closure)
    train_seconds = time.perf_counter() - t0
    with torch.no_grad():
        t1 = time.perf_counter()
        for _ in range(20):
            u = model()
        infer = (time.perf_counter() - t1) / 20
    return hist, u.detach().numpy().astype(np.float64), {"train_seconds": train_seconds, "inference_seconds": infer}


def oc_snapshots(iterations=(0, 2, 5, 10, 20, 40, 90)):
    """Re-run the Project 1 SIMP-OC loop and return the physical density at the
    requested update counts. Uses the P1 functions unchanged; the last snapshot
    must reproduce the saved P1 final design."""
    from top88 import (filter_matrix, finite_element_analysis,
                       optimality_criteria_update)
    st = Top88Settings()
    ke = element_stiffness(st.poisson)
    edof = element_dofs(st.nelx, st.nely)
    H, Hs = filter_matrix(st.nelx, st.nely, st.rmin)
    force, free = half_mbb_load_and_supports(st.nelx, st.nely)
    x = np.full(st.nelx * st.nely, st.volfrac)
    phys = x.copy()
    _, ee, c = finite_element_analysis(phys, st, edof, ke, force, free)
    out = {}
    for k in range(0, max(iterations) + 1):
        if k in iterations:
            out[k] = {"rho": phys.copy(), "compliance": c,
                      "gray_fraction": float(np.mean((phys > 0.1) & (phys < 0.9)))}
        if k == max(iterations):
            break
        dc = -st.penal * (st.young_solid - st.young_void) * phys ** (st.penal - 1.0) * ee
        dc = H.dot(x * dc) / (Hs * np.maximum(1.0e-3, x))
        x, phys = optimality_criteria_update(x, dc, np.ones_like(x), st, H, Hs)
        _, ee, c = finite_element_analysis(phys, st, edof, ke, force, free)
    return out
