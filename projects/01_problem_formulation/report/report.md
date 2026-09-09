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

The domain is divided into 120 by 40 equal-size, four-node finite elements. Each element has a relative density, giving the design vector

$$
\boldsymbol{\rho}=[\rho_1,\rho_2,\ldots,\rho_N]^{\mathsf T},\qquad N=4800,\qquad 0\leq\rho_e\leq1.
$$

The densities are dimensionless continuous variables. Zero represents void, one represents solid material, and intermediate values are allowed during optimization. The displacement vector is a state variable computed by finite-element analysis (FEM) for each density distribution.

| Quantity | Meaning | Value |
|---|---|---|
| $\boldsymbol{\rho}$ | Design variables | 4,800 element densities |
| $\mathbf{u}$ | Nodal displacement vector | Computed from equilibrium |
| $\mathbf{F}$ | Applied load vector | Downward unit load at the upper-left node |
| $f_v$ | Material volume limit | 0.50 |
| $E_0$, $E_{\min}$ | Solid and void moduli | 1 and $10^{-9}$ |
| $p$ | SIMP penalty exponent | 3 |
| $\nu$ | Poisson ratio | 0.30 |
| $r_{\min}$ | Sensitivity-filter radius | 3.5 element widths |

The calculation uses the normalized units of top88, with unit element side length and thickness. Displacement and compliance are reported in normalized units; their dimensional units are length and force times length, respectively.

## 3. Objective and Complete Optimization Formulation

The objective is to minimize compliance, which measures how much the structure deforms under the prescribed load:

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

Here, the subscript `f` denotes free displacement degrees of freedom, and `K_ff` is the corresponding stiffness matrix. The matrix is assembled from the element stiffnesses using the SIMP (Solid Isotropic Material with Penalization) relation

$$
E_e(\rho_e)=E_{\min}+\rho_e^p(E_0-E_{\min}).
$$

The cubic penalty makes intermediate densities less efficient: a density of 0.5 gives approximately 0.125 of the solid stiffness. This encourages material to concentrate into clear structural members. The small positive void modulus keeps the constrained system solvable as elements approach zero density.

## 4. Physical Meaning of the Objective and Constraints

- **Objective:** lower compliance means greater stiffness under the prescribed load. For this unit load, compliance equals the downward displacement at the loaded node.
- **Equilibrium equality:** every proposed material layout must satisfy static force balance.
- **Volume inequality:** at most half of the available material may be used. For equal-size elements, the average density is the material volume fraction.
- **Density bounds:** each cell can range from void to solid material.
- **Boundary equalities:** the left edge can move vertically but is restrained horizontally; the lower-right node can move horizontally but is restrained vertically.

Since adding material generally increases stiffness, the final design is expected to use nearly all of the allowed material.

## 5. Problem Classification and Source of Nonconvexity

This is a continuous, deterministic, single-objective nonlinear program (NLP). The design variables are continuous densities, the loads and material properties are fixed, and compliance is the only objective.

Although the FEM analysis is linear for a fixed density field, compliance depends nonlinearly on density through the stiffness matrix:

$$
C(\boldsymbol{\rho})=\mathbf{F}_f^{\mathsf T}\mathbf{K}_{ff}(\boldsymbol{\rho})^{-1}\mathbf{F}_f.
$$

The volume constraint and density bounds define a convex set in density space. Cubic SIMP interpolation favors concentrated material along competing load paths and produces a nonconvex objective, as the following numerical check illustrates.

On the same mesh, let design A alternate between densities 0.6 and 0.4 in a checkerboard pattern, and let design B swap these values. Both use 50% material. Their average is the uniform density-0.5 design. Evaluating all three designs under the same load and boundary conditions gives

$$
C(\boldsymbol{\rho}^A)=941.9618,\qquad C(\boldsymbol{\rho}^B)=981.9647.
$$

$$
C\left(\frac{\boldsymbol{\rho}^A+\boldsymbol{\rho}^B}{2}\right)=1026.8431>\frac{C(\boldsymbol{\rho}^A)+C(\boldsymbol{\rho}^B)}{2}=961.9633.
$$

For a convex objective, the average design would have compliance no greater than the average of the two original values. This result violates that condition and demonstrates nonconvexity. The calculation is included in the [verification script](../src/verify_top88.py).

Different starting layouts can lead to different local solutions. The OC method used here does not guarantee a global optimum.

## 6. Assumptions and Scope

The model assumes a homogeneous, isotropic, linear-elastic material, small deformation, and one static load case. A two-dimensional plane-stress model with uniform thickness represents the beam.

The study focuses on stiffness and material use. Stress limits, buckling, fatigue, and manufacturing requirements are omitted and would need to be considered in a practical component design.

## 7. Computational Solution and Results

### 7.1 Solution methodology

The [Python implementation](../src/top88.py) follows the classic 88-line method of Andreassen et al., using the official DTU example parameters:

~~~text
top88(120, 40, 0.5, 3.0, 3.5, 1)
~~~

The final argument selects sensitivity filtering. The calculation proceeds as follows:

1. Initialize every element at density 0.50.
2. Solve the FEM equilibrium equations and calculate compliance.
3. Calculate sensitivities, which indicate where adding material would most improve stiffness.
4. Smooth the sensitivities over neighboring elements to reduce checkerboard patterns.
5. Use the Optimality Criteria (OC) update to redistribute material within the volume limit, then repeat.

Each density can change by at most 0.2 per update. Iteration stops when the largest change between successive designs is at most 0.01:

$$
\Delta\rho_{\max}^{k}=\max_e|\rho_e^{k}-\rho_e^{k-1}|\leq0.01.
$$

### 7.2 Numerical results

| Quantity | Result |
|---|---:|
| Mesh | 120 by 40 elements |
| Design variables | 4800 |
| Iterations | 90 |
| Initial normalized compliance | 1026.8431 |
| Final normalized compliance | 210.0406 |
| Compliance reduction | 79.55% |
| Target volume fraction | 0.500000 |
| Final volume fraction | 0.499938 |
| Final maximum density change | 0.009922 |

![Optimized half MBB topology](../figures/final_topology.png)

The optimized beam has upper and lower members connected by diagonal load paths. Compared with the uniform density-0.5 initial design, compliance decreases by 79.55% while using essentially the same amount of material. Gray regions along the member boundaries represent intermediate densities.

![Compliance, volume fraction, and stopping-metric histories](../figures/convergence.png)

Compliance decreases rapidly during the first 20 updates and then levels off. The volume fraction remains close to 0.50, with small deviations caused by the numerical tolerance in the OC update. The final design satisfies the volume limit, and the maximum density change reaches 0.009922 at update 90, meeting the stopping criterion.

The result demonstrates how redistributing a fixed material budget can substantially improve structural stiffness.

### 7.3 Code and reproducibility

Run the following commands from the repository root:

~~~bash
pip install -r projects/01_problem_formulation/requirements.txt
python projects/01_problem_formulation/src/top88.py
~~~

The script generates the figures and saves the [iteration history](../results/history.csv), [final density field](../results/final_density.csv), and [run summary](../results/summary.json).

## References

1. MAE 598/494, [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html).
2. E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, “Efficient topology optimization in MATLAB using 88 lines of code,” *Structural and Multidisciplinary Optimization*, 43(1), 1–16, 2011. [doi:10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7).
3. DTU TopOpt, [Efficient topology optimization in MATLAB using 88 lines of code](https://www.topopt.mek.dtu.dk/apps-and-software/efficient-topology-optimization-in-matlab).
