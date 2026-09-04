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

Here, $\rho_e=0$ represents void and $\rho_e=1$ represents solid material. Intermediate values are permitted by the continuous relaxation but penalized by SIMP. The displacement vector $\mathbf{u}$ is a state variable: it is determined by the density field through finite-element equilibrium rather than selected independently by the designer.

| Quantity | Role | Value or dimension |
|---|---|---|
| $\boldsymbol{\rho}$ | Continuous design vector | 4,800 element densities |
| $\mathbf{u}$ | FEM displacement state | 9,922 nodal degrees of freedom |
| $\mathbf{F}$ | Prescribed load vector | Unit downward load at the upper-left node |
| $f_v$ | Maximum volume fraction | 0.50 |
| $E_0$, $E_{\min}$ | Solid and void moduli | 1 and $10^{-9}$ |
| $p$ | SIMP penalty exponent | 3 |
| $r_{\min}$ | Sensitivity-filter radius | 3.5 elements |

## 3. Objective and Complete Optimization Formulation

For a fixed load, minimizing compliance is equivalent to maximizing global stiffness. The complete finite-dimensional problem is

$$
\begin{aligned}
\underset{\boldsymbol{\rho},\mathbf{u}}{\operatorname{minimize}}\quad
& C(\boldsymbol{\rho},\mathbf{u})
=\mathbf{F}^{\mathsf T}\mathbf{u}
=\sum_{e=1}^{N}E_e(\rho_e)\,\mathbf{u}_e^{\mathsf T}\mathbf{k}_e^0\mathbf{u}_e \\
\operatorname{subject\ to}\quad
& \mathbf{K}(\boldsymbol{\rho})\mathbf{u}=\mathbf{F}, \\
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

In these expressions, $\mathbf{k}_e^0$ is the unit-modulus stiffness matrix of element $e$, $\mathbf{u}_e$ contains its eight displacement degrees of freedom, and $\mathbf{A}_e$ maps element quantities into the global system. The small positive value $E_{\min}$ prevents the stiffness matrix from becoming singular when an element approaches void.

## 4. Physical Meaning of the Objective and Constraints

- **Objective:** $C=\mathbf{F}^{\mathsf T}\mathbf{u}$ is structural compliance. For the normalized unit load used here, it is also the downward displacement at the loaded node. A smaller value therefore means a stiffer structure.
- **Equilibrium equality:** $\mathbf{K}(\boldsymbol{\rho})\mathbf{u}=\mathbf{F}$ requires the displacement field to satisfy static force balance for every proposed material layout.
- **Volume inequality:** the average density cannot exceed 0.50. Because all elements have equal area and thickness, average density is exactly the fraction of available material used.
- **Density bounds:** each element ranges from void to solid. SIMP keeps the problem continuous so that gradient-based optimization can be used.
- **Boundary equalities:** the left edge cannot move horizontally but may slide vertically; the lower-right node cannot move vertically but may slide horizontally. These are the same degrees of freedom shown in the figure and enforced in the code.

The volume constraint is expected to be active at the solution: with no competing penalty on material use, adding material generally increases stiffness. The computed final volume fraction of 0.499938 confirms this behavior.

## 5. Problem Classification and Source of Nonconvexity

The discretized problem is **continuous, deterministic, single-objective, high-dimensional, constrained, nonlinear, nonconvex, and finite-element-equilibrium constrained**.

Most of these labels follow directly from the model: the 4,800 densities are continuous variables; all loads and parameters are fixed; there is one compliance objective; and every design must satisfy equilibrium, volume, boundary, and box constraints.

The reason for calling the problem **nonconvex** is more specific than simply saying that it is nonlinear:

1. With $p=3$, element stiffness varies as $\rho_e^3$, not linearly with density. Doubling an intermediate density can therefore increase its stiffness by roughly a factor of eight before accounting for $E_{\min}$.
2. Eliminating the displacement state gives

   $$
   C(\boldsymbol{\rho})=\mathbf{F}^{\mathsf T}\mathbf{K}(\boldsymbol{\rho})^{-1}\mathbf{F}.
   $$

   Changing one density alters the global stiffness matrix and redistributes the displacement and strain energy throughout the entire structure. The element contributions are therefore coupled rather than independent.
3. The cubic penalty creates competing load paths. If one diagonal region becomes slightly denser, it becomes disproportionately stiffer, attracts more load, and may be retained while another plausible path disappears. Different paths can produce distinct locally optimal topologies with similar compliance.

The volume constraint and density bounds themselves are linear and define a convex feasible set. The nonconvexity arises primarily from the penalized density-stiffness relation combined with structural equilibrium. For comparison, the unpenalized case $p=1$ has an affine stiffness interpolation and a much more benign convex compliance structure under the same linear constraints. With $p=3$, the Optimality Criteria method is a local, gradient-based method: it can find a stationary design, but it does not certify the global optimum. Mesh resolution, filter radius, symmetry, continuation, and initialization can influence which local layout is obtained.

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
7. Repeat until the largest density change is no greater than 0.01.

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
| Measured runtime in the recorded run | approximately 6.6 s |

![Optimized half MBB topology](../figures/final_topology.png)

The optimized design forms a set of truss-like upper, lower, and diagonal load paths. Material is removed from regions that contribute little to transferring the applied load to the supports. The volume constraint is satisfied and essentially active: the final material fraction differs from the 0.50 target by about 0.0062 percentage points.

The nonzero gray fraction is expected because this educational configuration uses a sensitivity filter but no Heaviside projection. The transition bands also reflect the relatively large filter radius of 3.5 elements. A projection method could produce a sharper solid-void design but would change the selected formulation and introduce additional continuation parameters.

![Compliance, volume fraction, and stopping-metric histories](../figures/convergence.png)

Compliance decreases rapidly during the first 20 iterations and then approaches a plateau. Small late-stage oscillations arise from the filtered sensitivities and OC updates. The maximum density change eventually falls below the specified tolerance while the volume remains at its active constraint.

### 7.3 Code and reproducibility

From the repository root, install the scientific Python dependencies and run:

~~~bash
pip install -r requirements.txt
python projects/01_problem_formulation/src/top88.py
~~~

The command regenerates all submitted computational artifacts:

- [Convergence history](../results/history.csv)
- [Final physical-density field](../results/final_density.csv)
- [Machine-readable run summary](../results/summary.json)
- [Problem setup](../figures/problem_setup.png)
- [Final topology](../figures/final_topology.png)
- [Convergence plots](../figures/convergence.png)

The implementation is deterministic and does not use random initialization. Runtime is hardware- and software-dependent, so it is reported only as a reproducibility record rather than an algorithmic performance claim.

## References

1. MAE 598/494, [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html).
2. E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” *Structural and Multidisciplinary Optimization*, 43(1), 1–16, 2011. [doi:10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7).
3. DTU TopOpt, [Efficient topology optimization in MATLAB using 88 lines of code](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab).
