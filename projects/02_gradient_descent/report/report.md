# Project 2: Ill-Conditioning in the MBB Beam Equilibrium Solve

**Kangzheng Liu · OptiForge · MAE 598/494 Design Optimization**

## 1. Problem Identification and Motivation

[Project 1](../../01_problem_formulation/report/report.md) optimized the material layout of a half MBB beam. Every design update needed the beam's displacement, which means solving the equilibrium equations $K\mathbf u=\mathbf F$. Project 1 used a sparse direct solver; with 9,880 unknowns this takes less than a tenth of a second.

At scale this changes. Practical topology optimization is done in 3D with millions of unknowns or more [3, 4]. Factorizing $K$ then costs too much memory and time, so large codes solve it iteratively with the conjugate gradient method (CG) [3]. Iterative solvers slow down when $K$ is ill-conditioned, and topology optimization makes it ill-conditioned in two ways: fine meshes, and the large stiffness gap between solid and void material. Since the solve repeats at every design update, its cost limits how large a design an engineer can optimize.

This project writes the equilibrium solve as an energy minimization, explains why it is ill-conditioned, measures how much this slows gradient descent (GD), and tests two remedies from the course: CG and Jacobi preconditioning. The beam, load, and supports are those of Project 1: a downward unit force at the upper-left corner, $u_x=0$ along the left (symmetry) edge, and $u_y=0$ at the lower-right corner.

![Beam geometry, load, and supports](../../01_problem_formulation/figures/problem_setup.png)

## 2. Formulation

The beam is 120 by 40 with unit thickness, in the normalized units of Project 1. As in top88 [5], it is divided into square, four-node plane-stress elements, with $n_x=3n_y$ and element size $h=40/n_y$.

| Quantity | Meaning and units | Values or bounds |
|---|---|---|
| $\mathbf u$ | Nodal displacements (decision variables); continuous; length | $2(n_x+1)(n_y+1)$ components; support constraints below |
| $\rho_e$ | Element material fraction, fixed during a solve; dimensionless | $0\leq\rho_e\leq1$ |
| $\mathbf F$ | Applied nodal forces; force | Downward unit load |
| $E_0,E_{\min}$ | Solid and void moduli; force per area | $E_0=1$, $E_{\min}=10^{-9}$ |
| $p,\nu$ | SIMP exponent and Poisson ratio; dimensionless | $p=3$, $\nu=0.3$ |

As in Project 1, element $e$ has modulus $E_e=E_{\min}+\rho_e^p(E_0-E_{\min})$, and the stiffness matrix is assembled from one unit-modulus element matrix $\mathbf k_0$:

```math
K=\sum_e E_e\,P_e^{\mathsf T}\mathbf k_0P_e,
```

where $P_e$ selects the element's eight displacement components. The displacement minimizes total potential energy (strain energy minus the work of the load):

```math
\begin{aligned}
\min_{\mathbf u}\quad &\Pi(\mathbf u)=\frac12\mathbf u^{\mathsf T}K\mathbf u-\mathbf F^{\mathsf T}\mathbf u,\\
\text{s.t.}\quad &u_x=0\quad\text{on the left edge},\\
&u_y=0\quad\text{at the lower-right corner}.
\end{aligned}
```

Removing the fixed displacements leaves the free displacements $\mathbf u_f$, with no bounds. Their gradient and Hessian are

```math
\nabla\Pi=K_{ff}\mathbf u_f-\mathbf F_f,
\qquad H=K_{ff}.
```

Setting the gradient to zero gives equilibrium, $K_{ff}\mathbf u_f=\mathbf F_f$. $K_{ff}$ is symmetric positive definite because every $E_e>0$ (the reason $E_{\min}$ is not zero) and the supports remove all rigid-body motion. The problem is therefore an **unconstrained, continuous, deterministic, strictly convex quadratic program** with a unique solution. It has $2(n_x+1)(n_y+1)-(n_y+2)$ variables, or 9,880 on the Project 1 mesh. Because $\Pi$ is quadratic, the Hessian is the same everywhere, so the quadratic-bowl picture of conditioning is exact.

Below, $\mathbf u_k$ is the $k$-th iterate of $\mathbf u_f$, $\mathbf u^\star$ the solution, and $\mathbf e_k=\mathbf u_k-\mathbf u^\star$ the error.

## 3. Ill-Conditioning Mechanism

Each eigenvector $\mathbf v$ of $H$ is a deformation shape, and its eigenvalue is the stiffness of that shape:

```math
\lambda=\frac{\mathbf v^{\mathsf T}H\mathbf v}{\mathbf v^{\mathsf T}\mathbf v}.
```

The condition number $\kappa=\lambda_{\max}/\lambda_{\min}$ compares the stiffest and softest shapes. The main mechanism is **family B, a discretized differential operator**, with the mesh resolution $n_y$ as the knob. The figure shows the two extreme shapes on the 24 by 8 mesh.

![Softest and stiffest eigenvectors on the 24 by 8 mesh](../figures/extreme_modes.png)

- **Stiffest shape:** neighboring nodes move in opposite directions, so the energy sits inside single elements. In 2D the element matrix does not depend on element size (strains scale as $1/h$, element area as $h^2$), so $\lambda_{\max}$ stays nearly constant under refinement.
- **Softest shape:** the whole beam deflects on its support. For a fixed smooth shape, $\mathbf v^{\mathsf T}H\mathbf v$ does not change with the mesh, but $\mathbf v^{\mathsf T}\mathbf v$ sums over all nodes and grows as $1/h^2$. So $\lambda_{\min}$ falls as $h^2$, and $\kappa$ grows as $h^{-2}\propto n_y^2$. The code checks this with the trial shape $v_x=0$, $v_y=1-x/120$, whose energy is the same on every mesh.

**D2, growth under refinement.** We set every density to 0.5 and refine the mesh with the beam size fixed.

| Mesh | Variables | $\lambda_{\min}$ | $\lambda_{\max}$ | $\kappa$ | $\kappa$ after Jacobi scaling |
|---|---:|---:|---:|---:|---:|
| 12 by 4 | 124 | $3.26\times10^{-5}$ | 0.507 | 15,543 | 12,446 |
| 24 by 8 | 440 | $9.06\times10^{-6}$ | 0.537 | 59,263 | 51,807 |
| 48 by 16 | 1,648 | $2.39\times10^{-6}$ | 0.546 | 228,461 | 212,348 |
| 120 by 40 | 9,880 | $3.91\times10^{-7}$ | 0.549 | 1,404,156 | 1,363,443 |

$\lambda_{\max}$ barely moves while $\lambda_{\min}$ falls as $h^2$. The fitted slope of $\log\kappa$ against $\log n_y$ is 1.96, close to 2, the same rate as the 1-D Poisson example in the assignment.

![Condition number versus mesh resolution, before and after Jacobi scaling](../figures/mesh_conditioning.png)

**D2, survival under diagonal rescaling.** With

```math
D=\mathrm{diag}(H),\qquad J=D^{-1/2}HD^{-1/2},
```

each diagonal entry on a uniform mesh is one element value times the number of elements at that node (1, 2, or 4). The diagonal varies by at most a factor of 4, so $\kappa(J)\geq\kappa(H)/4$. The variables already have nearly the same scale; the two extreme shapes differ in wavelength, which a diagonal scaling cannot change. Measured: scaling lowers $\kappa$ by only 3–20%, and the slope stays near 2. The ill-conditioning is intrinsic.

**D1, spectrum.** The full spectra of the two smaller meshes show that only a few eigenvalues are very small; the softest mode sits well below the rest. Jacobi scaling shifts the spectrum up without changing its shape.

![Eigenvalue spectra for the two smaller meshes](../figures/eigenvalue_spectra.png)

The optimized Project 1 layout adds a second source, solid–void stiffness contrast (family A). It is treated in Section 5.2.

## 4. Effect of Ill-Conditioning

GD with the best constant step is

```math
\mathbf u_{k+1}=\mathbf u_k-\alpha(K_{ff}\mathbf u_k-\mathbf F_f),
\qquad \alpha=\frac{2}{\lambda_{\max}+\lambda_{\min}}.
```

Each update multiplies the error along eigenvector $i$ by $1-\alpha\lambda_i$. The stiffest mode limits the step, and the softest mode then shrinks by only $q=(\kappa-1)/(\kappa+1)=1-2/(\kappa+1)$ per update. This is the optimal-step counterpart of the rate $(1-\kappa^{-1})^k$ in the [first gradient-descent lecture](https://designinformaticslab.github.io/DesignOptimization2025/gradient_descent_pt1_2025.html), with $L=\lambda_{\max}$ and $\mu=\lambda_{\min}$. Either way, the number of updates grows in proportion to $\kappa$.

All methods start from zero and stop when the relative residual, which is also the normalized gradient norm, reaches

```math
r_k=\frac{\lVert K_{ff}\mathbf u_k-\mathbf F_f\rVert_2}{\lVert\mathbf F_f\rVert_2}\leq10^{-6}.
```

**D3, iterations to tolerance.** Late in the run, the residual lies almost entirely along the softest mode $\mathbf v_1$, so $r_k\approx c_1q^k$ with $c_1=|\mathbf v_1^{\mathsf T}\mathbf F_f|/\lVert\mathbf F_f\rVert_2$. This predicts $k\approx(\kappa/2)\ln(c_1/10^{-6})$ updates.

| Mesh | $\kappa$ | Predicted GD updates | Measured GD updates |
|---|---:|---:|---:|
| 12 by 4 | 15,543 | 93,647 | 93,648 |
| 24 by 8 | 59,263 | 338,559 | 338,559 |
| 48 by 16 | 228,461 | 1,229,961 | 1,229,961 |
| 120 by 40 | 1,404,156 | 6,929,438 | Not run |

Each mesh doubling costs GD about 3.6 times more updates, close to the growth of $\kappa$. On the Project 1 mesh GD would need about 6.9 million updates.

![GD and CG convergence on the 24 by 8 mesh](../figures/convergence_uniform_24x8.png)

**D3, convergence curves.** Panel (a) shows the residual on the 24 by 8 mesh. It drops quickly to about 0.09, then decays along $c_1q^k$ (dotted line) for over 300,000 updates. Panel (b) adds the relative objective gap $g_k=\big(\Pi(\mathbf u_k)-\Pi(\mathbf u^\star)\big)/\big(\Pi(\mathbf 0)-\Pi(\mathbf u^\star)\big)$. After 500 updates the residual is 0.098 but $g_k$ is still 0.92: a small residual along a soft mode hides a large displacement error.

Jacobi-scaled GD saves only 1.24, 1.14, and 1.07 times the GD updates on the three smaller meshes, matching the small drops in $\kappa$ from Section 3.

## 5. Proposed Solution and Demonstration

The [second gradient-descent lecture](https://designinformaticslab.github.io/DesignOptimization2025/gradient_descent_pt2_2025.html) writes preconditioned steepest descent as $\mathbf d_k=-M^{-1}\nabla\Pi(\mathbf u_k)$. $M=I$ is GD. $M=H$ is Newton's method, which solves a quadratic in one step; here that step is exactly the Project 1 direct solve, at the cost of factorizing $K_{ff}$. We use two cheaper remedies between these extremes, one for each mechanism.

### 5.1 Mesh refinement: conjugate gradient

Diagonal scaling cannot help with the mesh part, so the remedy has to change how search directions are combined. CG, also covered in that lecture, makes each new direction conjugate with respect to $K_{ff}$ to all earlier ones, so it never undoes earlier progress. Its standard error bound depends on $\sqrt\kappa$ rather than $\kappa$ [6]:

```math
\lVert\mathbf e_k\rVert_{K}\leq2\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^k\lVert\mathbf e_0\rVert_{K},
\qquad \lVert\mathbf e\rVert_K=\sqrt{\mathbf e^{\mathsf T}K_{ff}\mathbf e}.
```

Since $\kappa\propto n_y^2$, this bound suggests CG updates growing roughly as $n_y$. It bounds the energy error rather than our residual test, so it indicates a trend, not an exact count.

**D4, before and after.**

| Mesh | GD updates | CG updates | CG updates / $\sqrt\kappa$ |
|---|---:|---:|---:|
| 12 by 4 | 93,648 | 76 | 0.61 |
| 24 by 8 | 338,559 | 150 | 0.62 |
| 48 by 16 | 1,229,961 | 293 | 0.61 |
| 120 by 40 | About 6.9 million (predicted) | 710 | 0.60 |

CG updates double when the mesh doubles, while GD updates grow 3.6 times: the effective rate changes from $\kappa$ to $\sqrt\kappa$. The convergence figure in Section 4 shows both methods on the 24 by 8 mesh. CG's residual is not monotone because CG minimizes the energy error, and its objective gap falls at every update.

CG still slows down as the mesh is refined. Mesh-independent counts need a preconditioner that handles smooth modes on coarser grids, such as multigrid, which is the assignment's advanced remedy and what large topology optimization codes use [3].

### 5.2 Material contrast: Jacobi preconditioning

On the optimized Project 1 layout, nodes attached only to near-void elements have stiffness proportional to $E_{\min}$. The softest modes become deformations of the void material, $\lambda_{\min}\approx3.5\times10^{-3}E_{\min}$ for $E_{\min}\leq10^{-6}$, and $\kappa$ reaches $1.19\times10^{12}$. This is family A (multiscale stiffness).

This source shows up in the diagonal: a void node's diagonal entry is also proportional to $E_{\min}$. Jacobi preconditioning, $M=D$, is equivalent to CG on the symmetrically scaled matrix $J$ of Section 3. It brings the void modes back to the scale of the solid. Panel (a) below shows that $\kappa(J)$ is $1.13\times10^6$ for every $E_{\min}$, close to the uniform 120 by 40 mesh ($1.36\times10^6$, dotted line). For this layout, Jacobi scaling removes the contrast part and leaves the mesh part. Jacobi-preconditioned CG (Jacobi-PCG) therefore combines the two remedies.

![Condition numbers and convergence on the optimized layout](../figures/optimized_jacobi_fix.png)

**D4, before and after.**

| 120 by 40 layout | $\kappa$ | $\kappa$ after Jacobi | CG updates | Jacobi-PCG updates |
|---|---:|---:|---:|---:|
| Uniform (mesh only) | $1.40\times10^6$ | $1.36\times10^6$ | 710 | 686 |
| Project 1, $E_{\min}=10^{-9}$ (mesh and contrast) | $1.19\times10^{12}$ | $1.13\times10^6$ | Not converged after 5,000 ($r=7.2\times10^{-3}$) | 1,201 |

The same preconditioner saves 3% on the uniform mesh and turns a failed solve into a 1,201-update solve on the optimized layout. Its count barely depends on the contrast: 1,229, 1,216, and 1,201 updates for $E_{\min}=10^{-3}$, $10^{-6}$, and $10^{-9}$.

### 5.3 Check against Project 1

The Project 1 direct solve is the reference; CG gets the same 1,201 updates as Jacobi-PCG.

![Displacement and solver error on the optimized beam](../figures/project1_project2_comparison.png)

| Method | Updates | Relative residual | Compliance | Displacement error on the structure |
|---|---:|---:|---:|---:|
| Project 1: direct solve | — | $3.48\times10^{-12}$ | 210.040571 | Reference |
| CG | 1,201 | $9.51\times10^{-2}$ | 206.318024 | $2.9\times10^{-2}$ |
| Jacobi-PCG | 1,201 | $9.27\times10^{-7}$ | 210.040571 | $4.9\times10^{-10}$ |

Jacobi-PCG reproduces the Project 1 compliance to 11 significant digits, while CG with the same budget underestimates it by 1.8%. The error column uses nodes of elements with density at least 0.5, and the maps show those elements; near-void material is so soft that its displacement is poorly determined even by a converged solve. At this 2D size, the direct solve (Newton's one step) is still several times faster than Jacobi-PCG; iterative solvers pay off on the larger 3D problems in Section 1.

## 6. Assumptions and Simplifications

- Small-deformation, isotropic linear elasticity, plane stress, unit thickness, and one static load case.
- The density field is fixed during each solve, and every solve starts from zero. Inside an optimization loop, reusing the previous displacement would reduce the iteration count; this is not modeled.
- GD uses the best constant step, computed from the exact extreme eigenvalues. This favors GD; in practice those values are unknown.
- Performance is measured in updates; each GD, CG, or Jacobi-PCG update costs about one sparse matrix–vector product.
- The load magnitude does not affect $\kappa$ or the relative residuals, because $K$ does not depend on $\mathbf F$.

## Reproduction

Use Python 3.12 and run from the repository root:

```bash
python -m pip install -r projects/02_gradient_descent/requirements.txt
python projects/02_gradient_descent/src/conditioning_demo.py
```

The run takes about 2.5 minutes, mostly for GD. The [code](../src/conditioning_demo.py) reads the Project 1 FE model and saved density and uses a fixed random seed. The [results](../results/) contain condition numbers, solver histories, the Project 1 comparison, and environment metadata.

**Hand check.** Fixing all but the two upper vertical displacements of one solid square element gives

```math
H_{\mathrm{check}}=\frac1{91}
\begin{bmatrix}
45 & 5 \\
5 & 45
\end{bmatrix}.
```

Its eigenvalues are $40/91$ and $50/91$, so $\kappa=1.25$ and $q=1/9$. The load $(1,0)^{\mathsf T}$ splits equally between the two eigenvectors, so GD gives $r_k=9^{-k}$ and needs seven updates to reach $10^{-6}$. CG needs two, since it finishes in at most as many updates as there are unknowns. The code reproduces both counts; [verification.json](../results/verification.json) records this and the other checks.

## References

1. MAE 598/494, [Project 2: Ill-Conditioned Optimization](https://designinformaticslab.github.io/DesignOptimization2025/project2.html).
2. Kangzheng Liu, [Project 1: SIMP–OC Topology Optimization](../../01_problem_formulation/report/report.md).
3. O. Amir, N. Aage, and B. S. Lazarov, “On multigrid-CG for efficient topology optimization,” *Structural and Multidisciplinary Optimization*, 49, 815–829, 2014. [doi:10.1007/s00158-013-1015-5](https://doi.org/10.1007/s00158-013-1015-5).
4. N. Aage, E. Andreassen, B. S. Lazarov, and O. Sigmund, “Giga-voxel computational morphogenesis for structural design,” *Nature*, 550, 84–86, 2017. [doi:10.1038/nature23911](https://doi.org/10.1038/nature23911).
5. E. Andreassen et al., “Efficient topology optimization in MATLAB using 88 lines of code,” *Structural and Multidisciplinary Optimization*, 43, 1–16, 2011. [doi:10.1007/s00158-010-0594-7](https://doi.org/10.1007/s00158-010-0594-7).
6. J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, Section 5.1.
