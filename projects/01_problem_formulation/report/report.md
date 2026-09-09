# Project 1: Formulation and Solution of a SIMP Topology-Optimization Problem

**Course:** MAE 598/494 Design Optimization  
**Team:** OptiForge  
**Member:** Kangzheng Liu  
**Official brief:** [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html)

## 1. Problem Identification and Motivation

Consider a lightweight structural member that must carry a downward load without deforming excessively. A structural designer would like to remove material to reduce mass and cost, but removing material also makes the member more flexible. The design question is therefore:

> **Where should a limited amount of material be placed so that the structure is as stiff as possible?**

The classical half Messerschmitt-Bolkow-Blohm (MBB) beam is used as a clean model of this decision. Imagine the rectangular domain as a blank plate divided into 4,800 small cells. Each cell may contain anything from no material to solid material, but the design may use only 50% of the full plate. A unit load pushes downward at the upper-left corner, the left edge is a symmetry boundary, and the lower-right corner is supported by a vertical roller.

![Half MBB design domain, load, and supports](../figures/problem_setup.png)

An effective design does not spread the material uniformly. Instead, it forms a few continuous, truss-like paths that carry the load to the constrained boundaries. This simplified benchmark captures the central material-allocation problem found in lightweight bridge members, machine frames, robotic supports, and aerospace structures. It is not intended to be a production-ready component; it isolates the stiffness-versus-mass trade-off so that the optimization formulation can be stated and verified clearly.

## 2. Decision Variables and Fixed Inputs

The domain is discretized into

$$
N=120\times40=4800
$$

equal-size, four-node finite elements. The primary design variable is the relative density of each element, collected in the vector

$$
\boldsymbol{\rho}=[\rho_1,\rho_2,\ldots,\rho_N]^{\mathsf T}\in\mathbb{R}^{4800}, \qquad 0\leq\rho_e\leq1.
$$

Each density is a **dimensionless, continuous material fraction**, not a mass density in kg/m³. Zero represents void and one represents solid material. Intermediate values are permitted by the continuous relaxation but penalized by SIMP (Solid Isotropic Material with Penalization). The displacement vector is a state variable: it is determined by the density field through finite-element equilibrium rather than selected independently by the designer.

| Quantity | Role | Value or dimension |
|---|---|---|
| $\boldsymbol{\rho}$ | Continuous design vector | 4,800 element densities |
| $\mathbf{u}$ | FEM displacement state | 9,922 nodal degrees of freedom |
| $\mathbf{F}$ | Prescribed load vector | Unit downward load at the upper-left node |
| $f_v$ | Maximum volume fraction | 0.50 |
| $E_0$, $E_{\min}$ | Solid and void moduli | 1 and $10^{-9}$ |
| $p$ | SIMP penalty exponent | 3 |
| $r_{\min}$ | Sensitivity-filter radius | 3.5 elements |
| $\nu$ | Poisson ratio | 0.30 (dimensionless) |
| $t$ | Uniform thickness | 1 (normalized) |

All computational quantities use the normalized units of the top88 benchmark: the element side length, thickness, solid modulus, and load magnitude are each one. Before normalization, displacement has units of length, force has units of force, elastic modulus has units of force per area, and compliance has units of force times length. The reported compliance is therefore **not a displacement in mm or an energy in joules**. The domain has normalized dimensions 120 by 40; there are 4,961 nodes, 9,922 displacement DOFs, 42 prescribed DOFs, and 9,880 free DOFs.

## 3. Objective and Complete Optimization Formulation

For a fixed load, minimizing compliance is equivalent to maximizing global stiffness. The complete finite-dimensional problem is

$$
\begin{aligned}
\min_{\boldsymbol{\rho},\mathbf{u}}\quad
& C(\boldsymbol{\rho},\mathbf{u})=\mathbf{F}^{\mathsf T}\mathbf{u} \\
\text{s.t.}\quad
& \mathbf{K}_{ff}(\boldsymbol{\rho})\mathbf{u}_f=\mathbf{F}_f, \\
& \frac{1}{N}\sum_{e=1}^{N}\rho_e\leq f_v=0.50, \\
& 0\leq\rho_e\leq1, \qquad e=1,\ldots,N, \\
& u_x=0 \quad \text{on the left symmetry boundary}, \\
& u_y=0 \quad \text{at the lower-right roller support}.
\end{aligned}
$$

The density-dependent material interpolation and assembled stiffness matrix are

$$
E_e(\rho_e)=E_{\min}+\rho_e^p(E_0-E_{\min}), \qquad
\mathbf{K}(\boldsymbol{\rho})=\sum_{e=1}^{N}\mathbf{A}_e^{\mathsf T}E_e(\rho_e)\mathbf{k}_e^0\mathbf{A}_e.
$$

The subscript `f` denotes free DOFs; the two boundary equalities set the remaining DOFs to zero. Equilibrium is imposed on the free DOFs, exactly as in the code. The full assembled stiffness matrix has rigid-body modes before the boundary conditions are imposed, so its inverse is not used.

In the assembly expression, the matrix `A_e` extracts the eight element DOFs from the full displacement vector; its transpose scatters element forces into the global vector. The matrix `k_e^0` is the 8-by-8 Q4 plane-stress stiffness for unit modulus, unit thickness, and Poisson ratio 0.30. The positive void modulus keeps the **constrained** stiffness matrix nonsingular even when some densities are zero; a separate positive density lower bound is therefore unnecessary here.

At equilibrium, the objective can also be evaluated from element energies:

$$
C=\mathbf{F}^{\mathsf T}\mathbf{u}=\mathbf{u}^{\mathsf T}\mathbf{K}\mathbf{u}=\sum_{e=1}^{N}E_e(\rho_e) \mathbf{u}_e^{\mathsf T}\mathbf{k}_e^0\mathbf{u}_e=2U.
$$

Here `U` is the elastic strain energy. Compliance equals **twice** the strain energy for a linear system loaded from zero. This identity uses the zero prescribed displacements: support reactions do no work.

## 4. Physical Meaning of the Objective and Constraints

- **Objective:** $C=\mathbf{F}^{\mathsf T}\mathbf{u}$ is structural compliance. For the normalized unit load used here, it is also the downward displacement at the loaded node. A smaller value therefore means a stiffer structure.
- **Equilibrium equality:** the reduced system requires static force balance at every free DOF. At prescribed DOFs, reaction forces balance the structure; they are outputs rather than additional prescribed loads.
- **Volume inequality:** the average density cannot exceed 0.50. Because all elements have equal area and thickness, average density is exactly the fraction of available material used.
- **Density bounds:** each element ranges from void to solid. SIMP keeps the problem continuous so that gradient-based optimization can be used.
- **Boundary equalities:** the left edge cannot move horizontally but may slide vertically; the lower-right node cannot move vertically but may slide horizontally. These are the same degrees of freedom shown in the figure and enforced in the code.

The volume constraint is expected to be active at the solution: with no competing penalty on material use, adding material generally increases stiffness. The computed final volume fraction of 0.499938 confirms this behavior.

The left boundary is a symmetry cut, not a clamp. Its horizontal reactions can form a couple that balances the moment of the applied load. The only vertical support reaction is at the lower-right roller; it is approximately +1, balancing the downward unit load. Reflecting this half-domain gives a full MBB beam of width 240 with a total central load of 2 in the same normalization.

## 5. Problem Classification and Source of Nonconvexity

The discretized problem is **continuous, deterministic, single-objective, high-dimensional, constrained, nonlinear, nonconvex, and finite-element-equilibrium constrained**.

Most of these labels follow directly from the model: the 4,800 densities are continuous variables; all loads and parameters are fixed; there is one compliance objective; and every design must satisfy equilibrium, volume, boundary, and box constraints.

The reason for calling the problem **nonconvex** is more specific than simply saying that it is nonlinear.

**Penalized stiffness.** With $p=3$, element stiffness varies as $\rho_e^3$, not linearly with density. Doubling an intermediate density can therefore increase its stiffness by roughly a factor of eight before accounting for the small void modulus.

**Global equilibrium coupling.** Eliminating the displacement state gives

$$
C(\boldsymbol{\rho})=\mathbf{F}_f^{\mathsf T}\mathbf{K}_{ff}(\boldsymbol{\rho})^{-1}\mathbf{F}_f.
$$

Changing one density alters the global stiffness matrix and redistributes the displacement and strain energy throughout the entire structure. The element contributions are therefore coupled rather than independent.

**Competing load paths.** The cubic penalty amplifies small differences between possible layouts. If one diagonal region becomes slightly denser, it becomes disproportionately stiffer, attracts more load, and may be retained while another plausible path disappears. Different paths can therefore produce distinct locally optimal topologies with similar compliance.

The volume constraint and density bounds define a convex feasible set **in density space after eliminating the state**. The corresponding reduced objective is nonlinear. An inverse stiffness matrix or global coupling alone does not prove nonconvexity: with affine stiffness interpolation (`p = 1`), the reduced compliance is convex over densities for which the constrained stiffness is positive definite. With cubic SIMP interpolation, that convexity is generally lost.

**A concrete check on this mesh.** Assign alternating cells densities 0.6 and 0.4 to form design A, and interchange them to form design B. Both use exactly 50% material; their midpoint is the uniform density-0.5 design. For zero-based column and row indices `i,j`, the construction and convexity test are

$$
\rho^A_{ij}=0.5+0.1(-1)^{i+j},\qquad \rho^B_{ij}=1-\rho^A_{ij}.
$$

$$
C\left(\frac{\boldsymbol{\rho}^A+\boldsymbol{\rho}^B}{2}\right)=1026.8431>\frac{941.9618+981.9647}{2}=961.9633.
$$

A convex objective would satisfy the opposite inequality. This numerical counterexample demonstrates nonconvexity of the stated discrete SIMP problem, rather than inferring it merely from nonlinearity. These checkerboard fields are a mathematical test of the objective, not proposed optimized structures. The sensitivity filter discourages such patterns during optimization; it is not an additional feasibility constraint in the formulation above.

Optimality Criteria (OC) supplies a local update rule. The reported run reaches its density-change stopping threshold; this does **not** certify a KKT stationary point or a global optimum. In particular, sensitivity filtering modifies the raw compliance derivative and is not generally the exact gradient of the stated unfiltered objective. Mesh resolution, filter radius, symmetry, and initialization can affect the resulting layout.

## 6. Assumptions and Scope

The model deliberately makes the following simplifications:

- homogeneous, isotropic, linear-elastic material;
- small strains and small displacements;
- one static load case;
- two-dimensional plane stress with normalized unit thickness;
- square four-node elements on a fixed design domain;
- normalized geometry, load, and elastic modulus;
- sensitivity filtering with radius 3.5 elements;
- no stress, buckling, fatigue, contact, manufacturing, uncertainty, or three-dimensional constraints.

The result should therefore be interpreted as a demonstration of optimal material placement and load-path formation, not as a final manufacturable design. A production component would require dimensional material data, multiple load cases, stress and stability constraints, manufacturing restrictions, and higher-fidelity validation.

## 7. Computational Solution and Results

### 7.1 Solution methodology

The solution follows the classic 88-line topology-optimization method of Andreassen et al. The official DTU example call is reproduced:

~~~text
top88(120, 40, 0.5, 3.0, 3.5, 1)
~~~

The final argument selects the sensitivity filter. The expanded [Python implementation](../src/top88.py) performs the following operations:

1. Precompute the element stiffness matrix, element-to-DOF map, and sparse filter matrix.
2. Initialize all densities to the allowed volume fraction of 0.50.
3. Assemble the sparse global stiffness matrix and solve the free displacement DOFs.
4. Compute compliance and analytical element sensitivities.
5. Apply the distance-weighted sensitivity filter.
6. Update the density variables with the move-limited Optimality Criteria method; use bisection on the Lagrange multiplier to enforce the volume constraint.
7. Evaluate the updated design and repeat until the largest density change is no greater than 0.01; a cap of 250 updates reports an error if this does not occur.

**Why the sensitivity is negative.** Let `d_e` denote the derivative of reduced compliance with respect to an element density. Differentiating the free equilibrium system with respect to that density gives

$$
\mathbf{K}_{ff}\frac{\partial\mathbf{u}_f}{\partial\rho_e}=-\frac{\partial\mathbf{K}_{ff}}{\partial\rho_e}\mathbf{u}_f.
$$

Because the load is fixed and the stiffness is symmetric, substituting equilibrium into the derivative of compliance yields

$$
d_e=\frac{\mathrm{d}C}{\mathrm{d}\rho_e}=-\mathbf{u}_f^{\mathsf T}\frac{\partial\mathbf{K}_{ff}}{\partial\rho_e}\mathbf{u}_f=-p(E_0-E_{\min})\rho_e^{p-1}\mathbf{u}_e^{\mathsf T}\mathbf{k}_e^0\mathbf{u}_e.
$$

This is a **total derivative**: it accounts for the change in the displacement solution. Simply differentiating the element-energy formula while holding displacement fixed would give the wrong sign. A more negative sensitivity identifies a larger predicted compliance reduction from adding material.

The selected sensitivity filter replaces this derivative by

$$
H_{ej}=\max(0,r_{\min}-\lVert\mathbf{z}_e-\mathbf{z}_j\rVert_2),\qquad \widehat d_e=\frac{\sum_jH_{ej}\rho_jd_j}{\max(10^{-3},\rho_e)\sum_jH_{ej}}.
$$

Here `z_e` is the element-center position measured in element widths. With filter type 1, the physical density equals the design density: it is the **sensitivity**, not the density field, that is filtered.

For the equivalent volume constraint written as a density sum, the volume derivative is one. At an interior stationary point the unfiltered condition is `d_e + lambda = 0`. The classical damped OC rule uses the filtered derivative in the move-limited update

$$
\rho_e^{k+1}=\mathrm{clip}\left(\rho_e^k\sqrt{\frac{-\widehat d_e}{\lambda}},\ \max(0,\rho_e^k-m),\ \min(1,\rho_e^k+m)\right),\qquad m=0.2.
$$

The clip operation restricts its first argument to the interval given by the second and third arguments. Bisection chooses the multiplier `lambda`; its relative bracket tolerance is 0.001, matching top88. This is distinct from the density-change stopping criterion

$$
\Delta\rho_{\max}^{k}=\max_e|\rho_e^{k}-\rho_e^{k-1}|\leq0.01.
$$

### 7.2 Numerical results

| Quantity | Result |
|---|---:|
| Mesh | 120 by 40 elements |
| Design variables | 4800 |
| Displacement DOFs | 9922 |
| Iterations | 90 |
| Initial normalized compliance | 1026.8431 |
| Final normalized compliance | 210.0406 |
| Compliance reduction | 79.55% |
| Target volume fraction | 0.500000 |
| Final volume fraction | 0.499938 |
| Final maximum density change | 0.009922 |
| Elements with `0.1 < rho < 0.9` | 24.77% |
| Measured runtime in the recorded run | approximately 10.5 s |

![Optimized half MBB topology](../figures/final_topology.png)

The optimized design forms a set of truss-like upper, lower, and diagonal load paths. Material is removed from regions that contribute little to transferring the applied load to the supports. The volume constraint is satisfied and essentially active: the final material fraction differs from the 0.50 target by about 0.0062 percentage points.

The 79.55% reduction is relative to the **uniform density-0.5 SIMP initial design**, which uses the same material budget. It is not a comparison with a fully solid beam. Likewise, using 50% material does not mean that exactly 50% of the elements are binary solid, since intermediate densities remain.

The nonzero gray fraction is expected because this educational configuration uses a sensitivity filter but no Heaviside projection. The transition bands also reflect the relatively large filter radius of 3.5 elements. A projection method could produce a sharper solid-void design but would change the selected formulation and introduce additional continuation parameters.

![Compliance, volume fraction, and stopping-metric histories](../figures/convergence.png)

The three panels separate the objective, material constraint, and stopping criterion. Compliance decreases rapidly during the first 20 updates and then approaches a plateau. The volume panel zooms in around 0.50: its range is 0.499718–0.500274. Some intermediate iterates slightly exceed the exact volume bound because of the OC multiplier-bisection tolerance; the final design is below the bound. The change panel uses a logarithmic scale and explicitly marks 0.01. The first 89 updates exceed this stopping threshold, and update 90 reaches 0.009922.

Every history row now refers to a single design state after `k` OC updates, including compliance, volume, and gray fraction. The 91 rows contain the initial state (`k = 0`) and 90 updated states; the initial change is `NaN` because no preceding update exists. This aligns the saved history with the plotted curves and final density. Re-evaluating updated states changes reporting, not the classical OC update sequence.

### 7.3 Code and reproducibility

From the repository root, install the scientific Python dependencies and run:

~~~bash
pip install -r projects/01_problem_formulation/requirements.txt
python projects/01_problem_formulation/src/top88.py
python projects/01_problem_formulation/src/verify_top88.py
~~~

The first script regenerates the optimization artifacts; the second runs independent checks and writes the verification record:

- [Convergence history](../results/history.csv)
- [Final physical-density field](../results/final_density.csv)
- [Machine-readable run summary](../results/summary.json)
- [Problem setup](../figures/problem_setup.png)
- [Final topology](../figures/final_topology.png)
- [Convergence plots](../figures/convergence.png)
- [Verification code](../src/verify_top88.py) and [numerical checks](../results/verification.json)

The implementation is deterministic and does not use random initialization. Runtime is hardware- and software-dependent, so it is reported only as a reproducibility record rather than an algorithmic performance claim.

### 7.4 Numerical verification

The final saved density was loaded and analyzed again. Independent 2-by-2 Gauss integration reproduces the closed-form Q4 stiffness to a maximum absolute difference of 1.67e-16. The relative free-DOF equilibrium residual is 3.48e-12, and external-work and element-energy compliance agree to a relative error of 3.78e-12. All prescribed displacements are zero, and the computed reactions satisfy global force and moment balance. On a separate 6-by-2 mesh, central finite differences verify the unfiltered analytical sensitivity to a relative error of 3.80e-9. Density bounds, final volume feasibility, move limits, stopping logic, and summary/CSV consistency also pass.

These checks verify the discrete mechanics, derivative, and recorded result. They do not establish mesh independence, a binary manufactured design, KKT stationarity, or global optimality. The benchmark load is a normalization for comparing layouts, not evidence that a dimensional component meets small-displacement, stress, or safety requirements.

## References

1. MAE 598/494, [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html).
2. E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” *Structural and Multidisciplinary Optimization*, 43(1), 1–16, 2011. [doi:10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7).
3. DTU TopOpt, [Efficient topology optimization in MATLAB using 88 lines of code](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab).
