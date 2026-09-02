# Project 1 - Topology Optimization Formulation Using SIMP

**Course:** MAE 598/494 Design Optimization  
**Team:** OptiForge  
**Member:** Kangzheng Liu  
**Status:** Draft  
**Official brief:** [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html)

> Drafting note: replace every bracketed prompt, confirm all numerical values, and remove this note before submission.

## 1. Problem Identification and Motivation

This project formulates the topology optimization of a lightweight load-bearing bracket. The bracket must transfer an applied mechanical load to a fixed support while using a limited amount of material.

[Specify the engineering application, stakeholder, design domain dimensions, material, load magnitude and direction, support conditions, and why the stiffness-mass trade-off matters.]

<!-- Add the final schematic here after creating it: ![Problem setup](../figures/problem_setup.png) -->

## 2. Decision Variables

Discretize the design domain into $N$ finite elements. The design variable for element $e$ is its dimensionless relative density,

$$
\rho_e \in [\rho_{\min},1], \qquad e=1,\ldots,N.
$$

Collect the variables as

$$
\boldsymbol{\rho}=[\rho_1,\rho_2,\ldots,\rho_N]^{\mathsf T}\in\mathbb{R}^{N}.
$$

| Symbol | Definition | Role and type | Units or dimensions | Bounds |
|---|---|---|---|---|
| $\rho_e$ | Relative density of element $e$ | Continuous design variable | Dimensionless | $\rho_{\min}\leq\rho_e\leq1$ |
| $\mathbf{u}$ | Global nodal displacement vector | FEM state variable, not a freely selected design variable | [length; state vector dimension] | Determined by equilibrium and boundary conditions |
| $\mathbf{F}$ | Applied global load vector | Fixed parameter | [force; vector dimension] | Prescribed |
| $v_e$ | Volume of element $e$ | Fixed parameter | [volume] | Positive |
| $f_v$ | Allowable volume fraction | Fixed parameter | Dimensionless | [insert selected value] |

Under the SIMP interpolation, the element Young's modulus is

$$
E_e(\rho_e)=E_{\min}+\rho_e^p(E_0-E_{\min}),
$$

where $E_0$ is the solid-material modulus, $E_{\min}>0$ prevents a singular stiffness matrix, and $p>1$ penalizes intermediate densities.

[State whether a density filter or projection is part of the formulation. If used, define the filtered or physical density explicitly and use it consistently below.]

## 3. Objective Function

The objective is to minimize structural compliance, equivalently maximize stiffness for the prescribed load:

$$
\min_{\boldsymbol{\rho}}\quad
C(\boldsymbol{\rho})
=
\mathbf{F}^{\mathsf T}\mathbf{u}(\boldsymbol{\rho})
=
\sum_{e=1}^{N}
\mathbf{u}_e^{\mathsf T}
\mathbf{k}_e(\rho_e)
\mathbf{u}_e.
$$

Here, $\mathbf{u}_e$ is the element displacement vector and $\mathbf{k}_e(\rho_e)$ is the density-dependent element stiffness matrix.

[State the physical units of compliance for the selected unit system and explain why lower compliance represents a better bracket.]

## 4. Constraints

### 4.1 Equality constraints

Finite-element equilibrium:

$$
\mathbf{K}(\boldsymbol{\rho})\mathbf{u}=\mathbf{F}.
$$

Essential boundary conditions:

$$
\mathbf{u}=\mathbf{0}
\qquad \text{on }\Gamma_D.
$$

The first equation enforces static force balance. The second constrains the bracket at the prescribed support boundary $\Gamma_D$.

### 4.2 Inequality constraint

The material volume must not exceed the specified fraction of the full design domain:

$$
g_V(\boldsymbol{\rho})
=
\frac{\sum_{e=1}^{N}v_e\rho_e}
{\sum_{e=1}^{N}v_e}
-f_v
\leq0.
$$

[Explain the engineering meaning of the chosen volume fraction.]

### 4.3 Variable bounds

$$
\rho_{\min}\leq\rho_e\leq1,
\qquad e=1,\ldots,N.
$$

The lower bound maintains numerical stability, while the upper bound represents solid material.

### 4.4 Complete formulation

$$
\begin{aligned}
\min_{\boldsymbol{\rho},\mathbf{u}}\quad&
\mathbf{F}^{\mathsf T}\mathbf{u} \\
\text{subject to}\quad&
\mathbf{K}(\boldsymbol{\rho})\mathbf{u}=\mathbf{F},\\
&
\mathbf{u}=\mathbf{0}\quad\text{on }\Gamma_D,\\
&
\frac{\sum_{e=1}^{N}v_e\rho_e}
{\sum_{e=1}^{N}v_e}
\leq f_v,\\
&
\rho_{\min}\leq\rho_e\leq1,
\quad e=1,\ldots,N.
\end{aligned}
$$

## 5. Problem Classification

The discretized SIMP formulation is a deterministic, single-objective, high-dimensional, continuous, equality- and inequality-constrained nonlinear optimization problem with an embedded finite-element equilibrium model.

[Justify each classification attribute explicitly.]

In particular, although the underlying elasticity analysis is linear in displacement, the reduced objective depends on

$$
\mathbf{u}(\boldsymbol{\rho})
=
\mathbf{K}(\boldsymbol{\rho})^{-1}\mathbf{F}.
$$

The SIMP interpolation with $p>1$ introduces a nonlinear density-stiffness relation and generally produces a nonconvex problem with multiple local optima. The original solid-void problem would use binary variables and be combinatorial; SIMP replaces it with a continuous relaxation.

## 6. Assumptions and Simplifications

State and discuss the consequences of each assumption. At minimum, confirm or revise:

- Linear-elastic, isotropic material behavior.
- Small strains and small displacements.
- Static loading.
- Two-dimensional plane-stress or plane-strain idealization.
- Fixed design domain, supports, and loading.
- Prescribed material volume fraction.
- Mesh and filter choices.
- Minimum density used for numerical stability.
- Manufacturing constraints, stress limits, fatigue, uncertainty, and three-dimensional effects omitted from the base model.

[Explain which real-world behaviors are not captured and how they could change the optimized design.]

## 7. Computational Solution and Results (Bonus)

### 7.1 Solution methodology

[Describe FEM assembly, sensitivity analysis, density filtering, the Optimality Criteria update, initialization, continuation strategy, stopping criterion, software, and hardware.]

### 7.2 Results and interpretation

Include, at minimum:

- Initial and final density fields.
- Compliance convergence history.
- Volume-fraction history.
- Final constraint satisfaction.
- Engineering interpretation of the load paths.
- Limitations or sensitivity to modeling choices.

### 7.3 Code and reproducibility

~~~bash
# Replace with the exact command or notebook order.
~~~

Link the implementation, parameter file, and committed outputs used in the report.

## References

1. [Add the official assignment page and topology-optimization references used.]
