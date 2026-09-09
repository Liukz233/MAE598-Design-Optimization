# Project 1: Formulation and Solution of a SIMP Topology-Optimization Problem

**Course:** MAE 598/494 Design Optimization  
**Team:** OptiForge  
**Member:** Kangzheng Liu  
**Official brief:** [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html)

## 1. Problem Identification and Motivation

Lightweight structural design requires a balance between material use and stiffness. For a structural engineer, the task is to distribute a limited amount of material so that the member carries its load with minimal deformation.

> Where should a limited amount of material be placed to obtain the stiffest structure?

This project considers the classical half Messerschmitt-Bolkow-Blohm (MBB) beam. The rectangular design domain is divided into 4,800 cells, with material limited to 50% of the domain volume. A downward unit load acts at the upper-left corner. The left edge is a symmetry boundary, and a roller at the lower-right corner provides vertical support.

![Half MBB design domain, load, and supports](../figures/problem_setup.png)

The material distribution determines the load paths between the applied force and the supports. This design decision is relevant to lightweight bridge members, machine frames, and aerospace structures. The MBB benchmark provides a simple setting for studying how material placement affects stiffness.

## 2. Decision Variables and Fixed Inputs

The domain is discretized into

$$
N=120\times40=4800
$$

equal-size, four-node finite elements. The primary design variable is the relative density of each element, collected in the vector

$$
\boldsymbol{\rho}=[\rho_1,\rho_2,\ldots,\rho_N]^{\mathsf T}\in\mathbb{R}^{4800}, \qquad 0\leq\rho_e\leq1.
$$

The relative density is a dimensionless continuous variable, with zero representing void and one representing solid material. SIMP (Solid Isotropic Material with Penalization) penalizes intermediate densities. The displacement vector is the state variable obtained from finite-element equilibrium for each material distribution.

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

The model follows the normalized units of top88, with unit element side length, thickness, solid modulus, and load magnitude. Displacement and compliance are reported in these normalized units; their dimensional counterparts have units of length and force times length, respectively. The mesh contains 4,961 nodes and 9,922 displacement DOFs, of which 42 are prescribed and 9,880 are free.

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

The subscript `f` denotes the free DOFs. The remaining displacements are prescribed by the symmetry and support conditions.

The matrix `A_e` extracts the eight element DOFs from the global displacement vector, and its transpose assembles element forces. The matrix `k_e^0` is the 8-by-8 Q4 plane-stress stiffness matrix for unit modulus, unit thickness, and Poisson ratio 0.30. A small positive void modulus maintains the nonsingularity of the constrained system as densities approach zero.

At equilibrium, the objective can also be evaluated from element energies:

$$
C=\mathbf{F}^{\mathsf T}\mathbf{u}=\mathbf{u}^{\mathsf T}\mathbf{K}\mathbf{u}=\sum_{e=1}^{N}E_e(\rho_e) \mathbf{u}_e^{\mathsf T}\mathbf{k}_e^0\mathbf{u}_e=2U.
$$

Here `U` denotes elastic strain energy. With homogeneous displacement boundary conditions, the support reactions do no work and compliance equals twice the strain energy.

## 4. Physical Meaning of the Objective and Constraints

- **Objective:** compliance measures deformation under the prescribed load. For the unit load in this model, it equals the downward displacement at the loaded node.
- **Equilibrium equality:** the reduced system enforces force balance at the free DOFs, with support reactions recovered at the prescribed DOFs.
- **Volume inequality:** the average density gives the material volume fraction for equal-size elements and is limited to 0.50.
- **Density bounds:** each element has a relative density between zero and one.
- **Boundary equalities:** the left edge is restrained horizontally, and the lower-right node is restrained vertically.

Since additional material generally increases stiffness, the optimal design is expected to use the available material budget. The final volume fraction of 0.499938 is close to the prescribed limit.

Horizontal reactions along the symmetry boundary provide a couple to balance the applied moment. The lower-right roller carries a vertical reaction of approximately +1. Reflecting the half-domain gives a full beam of width 240 under a central load of 2 in the same normalization.

## 5. Problem Classification and Source of Nonconvexity

The formulation is a continuous, deterministic, single-objective nonconvex nonlinear program (NLP). Its 4,800 density variables are subject to equilibrium, volume, and bound constraints, with fixed loads and material parameters.

Eliminating the displacement state gives the reduced objective

$$
C(\boldsymbol{\rho})=\mathbf{F}_f^{\mathsf T}\mathbf{K}_{ff}(\boldsymbol{\rho})^{-1}\mathbf{F}_f.
$$

A change in density modifies the global stiffness matrix and redistributes displacement and strain energy. With cubic SIMP interpolation, denser regions gain stiffness disproportionately, favoring distinct load paths.

The volume constraint and density bounds define a convex set in density space. For affine stiffness interpolation (`p = 1`), the reduced compliance is convex wherever the constrained stiffness is positive definite. Cubic interpolation generally loses this property, as the following calculation demonstrates.

Consider two feasible density fields on the current mesh: design A alternates between 0.6 and 0.4, and design B interchanges these values. Both have a volume fraction of 0.50, and their midpoint is the uniform density-0.5 design. Using zero-based column and row indices `i,j`,

$$
\rho^A_{ij}=0.5+0.1(-1)^{i+j},\qquad \rho^B_{ij}=1-\rho^A_{ij}.
$$

$$
C\left(\frac{\boldsymbol{\rho}^A+\boldsymbol{\rho}^B}{2}\right)=1026.8431>\frac{941.9618+981.9647}{2}=961.9633.
$$

The midpoint compliance exceeds the average endpoint compliance, violating the convexity inequality. These fields therefore provide a numerical counterexample for the discrete SIMP objective. Sensitivity filtering discourages checkerboard patterns through the update rule while leaving the stated density constraints unchanged.

Optimality Criteria (OC) is used to obtain a local numerical solution. The sensitivity filter modifies the raw compliance derivative, and convergence is assessed by the density-change criterion. KKT stationarity and global optimality are not evaluated in this study.

## 6. Assumptions and Scope

The analysis assumes:

- homogeneous, isotropic, linear-elastic material;
- small strains and small displacements;
- one static load case;
- two-dimensional plane stress with normalized unit thickness;
- square four-node elements on a fixed design domain;
- normalized geometry, load, and elastic modulus;
- sensitivity filtering with radius 3.5 elements;
- no stress, buckling, fatigue, contact, manufacturing, uncertainty, or three-dimensional constraints.

The study focuses on stiffness under a single static load. Application to a manufactured component would require dimensional material and loading data, together with stress, stability, and manufacturing constraints. Mesh dependence is also outside the scope of this calculation.

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

Let `d_e` denote the compliance sensitivity to an element density. Differentiating the free equilibrium system gives

$$
\mathbf{K}_{ff}\frac{\partial\mathbf{u}_f}{\partial\rho_e}=-\frac{\partial\mathbf{K}_{ff}}{\partial\rho_e}\mathbf{u}_f.
$$

Because the load is fixed and the stiffness is symmetric, substituting equilibrium into the derivative of compliance yields

$$
d_e=\frac{\mathrm{d}C}{\mathrm{d}\rho_e}=-\mathbf{u}_f^{\mathsf T}\frac{\partial\mathbf{K}_{ff}}{\partial\rho_e}\mathbf{u}_f=-p(E_0-E_{\min})\rho_e^{p-1}\mathbf{u}_e^{\mathsf T}\mathbf{k}_e^0\mathbf{u}_e.
$$

The derivative includes the dependence of displacement on density. Its negative sign indicates that adding material reduces compliance; elements with more negative sensitivities offer a greater local benefit.

The selected sensitivity filter replaces this derivative by

$$
H_{ej}=\max(0,r_{\min}-\lVert\mathbf{z}_e-\mathbf{z}_j\rVert_2),\qquad \widehat d_e=\frac{\sum_jH_{ej}\rho_jd_j}{\max(10^{-3},\rho_e)\sum_jH_{ej}}.
$$

Here `z_e` is the element-center position measured in element widths. Filter type 1 smooths the sensitivities, and the physical density equals the design density.

Writing the volume constraint as a density sum gives a volume derivative of one. The interior stationarity condition is then `d_e + lambda = 0`. The damped OC update uses the filtered sensitivity:

$$
\rho_e^{k+1}=\mathrm{clip}\left(\rho_e^k\sqrt{\frac{-\widehat d_e}{\lambda}},\ \max(0,\rho_e^k-m),\ \min(1,\rho_e^k+m)\right),\qquad m=0.2.
$$

The clip operation enforces the density bounds and move limit. The multiplier `lambda` is found by bisection with a relative bracket tolerance of 0.001. Iteration stops when

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

The optimized topology contains upper and lower members connected by diagonal load paths. Material is concentrated along these paths, giving a final compliance of 210.0406 at a volume fraction of 0.499938.

Relative to the uniform density-0.5 initial design, compliance decreases by 79.55% at essentially the same material volume.

Intermediate densities occupy 24.77% of the elements, mainly along the transition bands. This configuration uses a sensitivity filter with radius 3.5 elements and no Heaviside projection.

![Compliance, volume fraction, and stopping-metric histories](../figures/convergence.png)

Compliance decreases rapidly during the first 20 updates and then approaches a plateau. The volume fraction ranges from 0.499718 to 0.500274 because of the multiplier-bisection tolerance, with the final design satisfying the volume bound. The maximum density change first falls below 0.01 at update 90, reaching 0.009922.

The history contains the initial state and 90 updated states. Each row records compliance, volume, and gray fraction for the same density field; the initial density change is marked `NaN`.

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

The calculation uses a uniform initial density and is deterministic. Software versions and the measured runtime are recorded in the run summary.

### 7.4 Numerical verification

Reanalysis of the saved density confirms the reported compliance. Independent 2-by-2 Gauss integration agrees with the closed-form Q4 stiffness to a maximum absolute difference of 1.67e-16. The relative free-DOF equilibrium residual is 3.48e-12, and the relative difference between external-work and element-energy compliance is 3.78e-12. The prescribed displacements and global force and moment balances are satisfied.

On a separate 6-by-2 mesh, central finite differences agree with the unfiltered analytical sensitivity to a relative error of 3.80e-9. Checks of density bounds, final volume, move limits, stopping logic, and consistency between the summary and CSV outputs also pass.

## References

1. MAE 598/494, [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html).
2. E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” *Structural and Multidisciplinary Optimization*, 43(1), 1–16, 2011. [doi:10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7).
3. DTU TopOpt, [Efficient topology optimization in MATLAB using 88 lines of code](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab).
