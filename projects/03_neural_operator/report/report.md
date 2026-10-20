# Project 3: Training a Physics-Informed Neural Parameterization for Topology Optimization

**Kangzheng Liu · OptiForge · MAE 598/494 Design Optimization**  
**Official brief:** [Project 3: PINN/PINO in the Field](https://designinformaticslab.github.io/DesignOptimization2025/project3.html)  
**Researcher:** Dr. Leixin Ma, OASIS Lab, ASU · **Path:** C (diagnosis and fix)

## Summary

Dr. Ma works on topology optimization (TO). Every design update in TO needs an equilibrium solve, and her group is exploring whether a neural network can represent the displacement field and be trained on the potential energy alone, without FE labels. She asked how accurate this is on real TO designs and what limits it.

I tested this on the Project 1 MBB beam, training a plain MLP on the exact FE energy of designs taken from the Project 1 optimization run.

- With Adam, the displacement error on the final design is 6.3–8.3% over three seeds, well above the 1% target, and it grows as the design matures.
- The same network fitted directly to the FE solution reaches 0.47%. The network can represent the answer; the difficulty is in optimizing the energy loss.
- The energy loss weights each error mode by its stiffness. The soft, smooth modes that carry most of the compliance are learned last, and Adam's per-parameter scaling does not help with them.
- Switching to L-BFGS after 10,000 Adam steps brings the error to 0.18–0.42%. In 2D the FE direct solve is still about 3,500 times faster, so the result concerns how to train these networks, not a speedup.

## 1. Outreach Log

| Date | Person | Format | Outcome |
|---|---|---|---|
| Fri, Oct 9, 2026, 1:00 pm | Dr. Leixin Ma, OASIS Lab | In person | Interview: her TO work, the cost of the equilibrium solve, and neural parameterization of it. Agreed on the question for this project. |
| Fri, Oct 16, 2026, 1:00 pm | Dr. Leixin Ma | In person | Follow-up on the minimal case and the experiment plan. |
| Tue, Oct 20, 2026 | Dr. Leixin Ma | Email | Problem brief and recommendation memo sent for review. |

The study uses only the public Project 1 and 2 code; no unpublished code or data were used.

## 2. Problem Brief

*Sent to Dr. Ma for review on Oct 20, 2026.*

**Problem and value.** TO decides where to place material so that a part is as stiff as possible for a given weight. It is used to design lightweight structures and architected materials. Each design update needs one equilibrium solve, and a run takes about 100 updates. In 2D this is cheap; the Project 1 beam solves in under 0.1 s. In 3D or with nonlinear materials, each solve is a large sparse factorization, often inside a Newton loop, and the solves dominate the cost of a design study.

**Governing PDE.** Static equilibrium with a density-dependent stiffness:

```math
\nabla\cdot\boldsymbol\sigma(\mathbf u)=\mathbf 0\ \text{in }\Omega,\qquad
\mathbf u=\mathbf 0\ \text{on }\Gamma_D,\qquad
\boldsymbol\sigma\mathbf n=\mathbf t\ \text{on }\Gamma_N,\qquad
\boldsymbol\sigma=E(\rho)\,\mathbb C:\boldsymbol\varepsilon(\mathbf u),
```

with the SIMP modulus $E(\rho)=E_{\min}+\rho^p(E_0-E_{\min})$. Equivalently, $\mathbf u$ minimizes the total potential energy $\Pi$. The physics is fully known; only $\rho$ changes from one solve to the next.

**Neural parameterization.** Her group is looking at representing the displacement by a network, $\mathbf u=\mathbf u_{\boldsymbol\theta}(\mathbf x)$, and training $\boldsymbol\theta$ by minimizing $\Pi$. Training needs no FE solutions, and a trained network could be reused across design updates. Such networks are usually trained with Adam. How close they get to FE on real TO designs, and what limits them, is the open question.

**Useful accuracy.** About 1% in compliance, the TO objective. Errors of a few percent can change which elements an OC update adds or removes.

## 3. Minimal Case Formulation

**Domain and discretization.** The setup is that of Projects 1 and 2 [2]: a $120\times40$ half MBB beam meshed with $120\times40$ four-node plane-stress elements, $E_0=1$, $E_{\min}=10^{-9}$, $p=3$, $\nu=0.3$, a downward unit load at the top-left node, $u_x=0$ on the left edge, and $u_y=0$ at the bottom-right node [3]. The FE displacement minimizes

```math
\Pi(\mathbf u)=\tfrac12\mathbf u^{\mathsf T}K(\boldsymbol\rho)\mathbf u-\mathbf F^{\mathsf T}\mathbf u .
```

**The PINN.** The network maps a node position to its displacement:

```math
u_x=u_c\,\frac{x}{L}\,N_x(\hat x,\hat y;\boldsymbol\theta),\qquad
u_y=u_c\,d(x,y)\,N_y(\hat x,\hat y;\boldsymbol\theta),
```

where $N$ is an MLP with four hidden layers of 64 tanh units, $(\hat x,\hat y)=(x/L,y/H)$, and $d$ is the distance to the roller (capped at 1). The factors $x/L$ and $d$ make the supports exact. The scale $u_c=108$ is the Euler–Bernoulli deflection of the solid beam and only normalizes the output. The network is evaluated at the 4,961 FE nodes, and the loss is the FE energy of those values, $L(\boldsymbol\theta)=\Pi(\mathbf u_{\boldsymbol\theta})$. With exact BCs,

```math
\Pi(\mathbf u)-\Pi(\mathbf u^\star)=\tfrac12\,\mathbf e^{\mathsf T}K\mathbf e,\qquad \mathbf e=\mathbf u-\mathbf u^\star ,
```

so the loss gap is the energy norm of the error and the loss is minimized exactly by the FE solution. Any difference from FE therefore comes from the network or its training, not from discretization.

This is the energy (Deep Ritz) form of a PINN [4, 5] rather than the strong-form residual used in the course notes [1]. The strong form fits TO poorly: it contains $\nabla E$, and $E$ changes by nine orders of magnitude across a few elements at every solid–void boundary. The energy form needs only first derivatives and handles these jumps naturally. It is also the form Dr. Ma's question is about.

**Training.** Adam for 20,000 steps, with the learning rate decaying from $10^{-3}$ to $10^{-4}$. The fix in Section 4.7 uses 10,000 Adam steps followed by 500 L-BFGS iterations [6]. Adam runs in float32, and each element's rigid translation is subtracted before its energy is computed to avoid round-off; L-BFGS runs in float64. Errors are relative $L^2$ errors on the structure (nodes touching an element with $\rho\ge0.5$), as in Project 2.

**Simplifications.** The case is 2D instead of 3D, linear instead of hyperelastic, and uses one small MLP per design instead of a network that takes the design as input. It keeps what the question is about: the energy loss, the optimizer, and real TO density fields with a $10^9$ stiffness contrast and slender members. Linear elasticity also makes the FE solution an exact reference.

**Difficulty knob.** Design maturity $k$, the number of OC updates in the Project 1 run. I solve on the designs after $k=0,2,5,10,20,90$ updates; $k=90$ is the final Project 1 design, reproduced exactly (compliance 210.0406). Two secondary knobs come from Project 2: the void modulus $E_{\min}$ and the mesh resolution $n_y$.

![Designs along the knob](../figures/knob_designs.png)

## 4. Experiments

### 4.1 P1 – Baseline

The baseline is the Project 1 sparse direct solve (SciPy, one CPU core). Its relative residual is below $6\times10^{-12}$ on every design.

| OC update $k$ | 0 | 2 | 5 | 10 | 20 | 90 |
|---|---:|---:|---:|---:|---:|---:|
| Compliance | 1026.8 | 416.6 | 314.7 | 275.3 | 214.5 | 210.0 |
| Gray fraction ($0.1<\rho<0.9$) | 100% | 99% | 79% | 66% | 26% | 25% |
| Solve time (median of 7) | 79 ms | 92 ms | 84 ms | 83 ms | 81 ms | 77 ms |

### 4.2 P2 – Loss components

![Loss components during training](../figures/loss_components.png)

With exact BCs the loss has a single term. Panels (a) and (b) split it into strain energy and load work and compare each with its FE value. Both decrease together, so there is no imbalance between them. Under Adam every curve falls roughly as $1/t$, which in practice looks like a plateau. The PINN compliance is below the FE value in every run with exact BCs, so an under-trained network makes the design look stiffer than it is.

Panels (c) and (d) add a penalty $\beta\sum u_{\text{fixed}}^2$ instead of exact BCs and log the two terms separately. With $\beta=0.1$ the penalty settles at about 10% of $|\Pi^\star|$ and the supports move by 2.4% of the peak displacement, so training converges to the wrong problem. With $\beta=10$ and $10^3$ the stiff penalty term slows the energy term down, and the error is still 29% and 95% after 20,000 steps. All other runs use exact BCs.

### 4.3 P3 – Error versus knob

![Error versus the three knobs](../figures/error_vs_knobs.png)

Relative displacement error on the structure (seed 0):

| OC update $k$ | 0 | 2 | 5 | 10 | 20 | 90 |
|---|---:|---:|---:|---:|---:|---:|
| Energy loss, Adam | 3.48% | 1.57% | 1.95% | 3.17% | 7.53% | 8.26% |
| Energy loss, Adam then L-BFGS | 0.004% | 0.045% | 0.048% | 0.14% | 0.80% | 0.32% |
| Supervised fit to FE, Adam | 0.24% | 0.24% | 0.37% | 0.39% | 0.49% | 0.47% |

With Adam the error is above 1% on every design and grows from 1.6% at $k=2$ to 8.3% at $k=90$. The uniform start, $k=0$, does not follow this trend. Three seeds on the final design give 6.3–8.3%, and doubling the budget to 40,000 steps still leaves 3.3%.

For the secondary knobs, lowering $E_{\min}$ from $10^{-1}$ to $10^{-9}$ raises $\kappa(K)$ from $2.0\times10^6$ to $1.2\times10^{12}$, but the Adam error only moves from 6.8% to 8.3%. Mesh refinement matters more. On the uniform design with the load spread over 10 units (to avoid the point-load singularity), going from $n_y=16$ to 80 raises $\kappa(K)$ 25 times and the Adam error 4.6 times, from 1.1% to 5.2%.

### 4.4 P4 – Where the error is

![Error maps and spectra](../figures/where_is_the_error.png)

The Adam error is smooth and spread over the whole structure: the network misses part of the global deflection. Along the beam axis, 89% of its power on the final design, and over 99% on the uniform design, lies in the four lowest modes, whose wavelengths are at least the beam height. This is not spectral bias, which would leave the error at high frequencies. The supervised fit shows the opposite pattern, with a patchy error and only 38% and 32% of its power in those modes.

Projected onto the eigenvectors of $K$ (uniform design), the five softest modes carry 95.6% of the compliance. They also hold 99.9% of the squared Adam error, but only 0.4% of the supervised error.

### 4.5 P5 – Cost

One core of a 2.1 GHz Intel Xeon, no GPU.

| | FE direct solve | PINN, Adam | PINN, Adam then L-BFGS |
|---|---:|---:|---:|
| Time per design | 0.077 s | 166 s training | 271 s training |
| Inference | — | 2.2 ms | 2.3 ms |
| Error, final design | residual $\sim10^{-12}$ | 8.3% | 0.32% |

A single-design PINN must be retrained at every design update, so it never pays off. For an operator trained once and reused, $N^\star=T_{\text{train}}/(T_{\text{sol}}-T_{\text{inf}})$. Even with training as cheap as one PINN here, $N^\star\approx3{,}600$ solves, against about 90 in one Project 1 run. To pay off within one 2D run, training would have to take under $90\times(0.077-0.002)\approx6.7$ s. In 2D the direct solve is the right tool.

### 4.6 Path B – Diagnosis

**Expressivity or optimization.** Following Krishnapriyan et al. [7], the same network with the same Adam budget is trained to fit the FE displacement directly. It reaches 0.24–0.49% on every design, 18 times lower than the energy-trained error on the final design. The network can represent the solution, so the failure lies in optimizing the energy loss.

**Mechanism.** The two losses differ only by $K$. With $J=\partial\mathbf u/\partial\boldsymbol\theta$ and $\Theta=JJ^{\mathsf T}$ the neural tangent kernel [8], small steps change the error as

```math
\dot{\mathbf e}=-\Theta K\mathbf e\quad\text{(energy loss)},\qquad \dot{\mathbf e}=-\Theta\,\mathbf e\quad\text{(supervised loss)}.
```

Under the energy loss, the error component along an eigenvector $\mathbf v_i$ of $K$ is driven by a gradient proportional to $\lambda_i$. The softest modes of an elastic structure are its smooth global deformations, and they carry most of the compliance; on the uniform design the softest mode alone carries 93% (Project 2). These modes are learned last, and an error $a\mathbf v_i$ left in them costs only $\tfrac12\lambda_i a^2$ in the loss. Section 4.4 shows exactly this: the energy-trained error sits in the softest modes, and the supervised error does not.

Adam scales each parameter by its own gradient history, which is a diagonal preconditioner in $\boldsymbol\theta$. A soft global mode involves all parameters at once, so diagonal scaling cannot single it out. This mirrors Project 2, where Jacobi scaling could not remove the mesh-induced soft mode but CG could.

**Project 2 mechanisms.** The solid–void contrast does not carry over. The void modes are soft, but the solution has almost no component on them, so the loss ignores them, which is harmless for compliance. Mesh refinement does carry over: a finer mesh lets the energy resolve finer features of the network output, which widens the range of stiffnesses training has to cover.

**Root cause.** The energy loss is ill-conditioned along the smooth, compliance-carrying modes of the structure, and Adam cannot precondition these directions.

### 4.7 Path C – Fix

The remedy has to act on curvature rather than on individual parameters. L-BFGS builds an approximation of the inverse Hessian, close to $(J^{\mathsf T}KJ)^{-1}$ near the solution, from recent gradients, and takes large steps along low-curvature directions. It plays the role CG played in Project 2. Adam runs first because L-BFGS works best near a minimum; the same combination is standard for residual-based PINNs [9].

Final design, three seeds, mean (range):

| | Adam, 20k steps | Adam 10k, then L-BFGS 500 |
|---|---:|---:|
| Displacement error | 7.0% (6.3–8.3%) | **0.31% (0.18–0.42%)** |
| Compliance error | 6.5% (5.9–7.7%) | **0.28% (0.18–0.40%)** |
| Loss gap $(\Pi-\Pi^\star)/\lvert\Pi^\star\rvert$ | 6.0% (5.3–7.2%) | 0.31% (0.22–0.40%) |
| Training time | 168 s | 271 s |

![Accuracy versus time and the cost of one solve](../figures/cost_and_fix.png)

Adam given more time (40,000 steps, 339 s) reaches only 3.3%, so the gain comes from the optimizer, not from the extra time. Along the knob, L-BFGS reduces the error by 9 to 900 times and keeps every design under 1% (Section 4.3). It also acts on the diagnosed modes: on the uniform design, the five softest modes held 99.9% of the Adam error, but only 12% of the L-BFGS error, which is 900 times smaller.

## 5. Recommendation Memo

> **To:** Dr. Leixin Ma, OASIS Lab · **From:** Kangzheng Liu · **Date:** Oct 20, 2026  
> **Re:** Accuracy of energy-trained neural parameterizations on TO designs
>
> **Question.** How accurately can a network trained only on the potential energy represent the displacement on real TO designs, and what limits it?
>
> **What I did.** A plain MLP trained on the exact FE energy of 2D MBB designs taken from a SIMP–OC run (linear elasticity, exact BCs), compared with the FE direct solve.
>
> **Findings.**
> 1. With Adam, the error on the final design is 6–8% and rises as the design matures. The same network fitted to FE data reaches 0.5%, so the limit is the optimizer on this loss, not the network.
> 2. The energy loss learns the soft, compliance-carrying modes last, and Adam cannot target them.
> 3. Running 10,000 Adam steps followed by L-BFGS brings the error to 0.2–0.4% in about 270 s.
> 4. An under-trained network underestimates compliance, and the loss value alone does not show by how much.
> 5. Penalty BCs add a separate failure that exact BCs avoid. In float32, each element's rigid translation should be removed before its energy is computed.
> 6. In 2D the FE solve takes 0.08 s. A speedup can only come from 3D, nonlinear problems, or reuse across designs.
>
> **Confidence.** High for 2D linear elasticity. Moderate for larger networks that take the design as input, and for hyperelastic materials, where the loss is nonconvex.
>
> **Recommendation.** Add an L-BFGS (or other curvature-aware) stage before changing the architecture, impose BCs exactly, and check compliance against FE on a few designs.
>
> **Next step.** Repeat the Adam versus Adam-plus-L-BFGS comparison with a design-conditioned network on one hyperelastic benchmark, and measure the cost per design update against FE.

## 6. Assumptions and Limitations

- **2D linear elasticity, one load case.** The mechanism does not depend on dimension, but the cost comparison does: in 3D the FE solve is far more expensive.
- **Hyperelastic materials were not tested.** The energy is then nonconvex, and L-BFGS line searches can fail.
- **One MLP per design.** A network that takes the design as input and is warm-started would start closer to the solution, but the conditioning of the loss is the same.
- **Output scale.** Results depend on $u_c$; scaling it by the design's mean modulus made the Adam error 1.9 to 5.6 times larger. All comparisons use the same FE-free scale.
- **Seeds.** Three seeds were run only on the final design (Adam 6.3–8.3%, L-BFGS 0.18–0.42%); other points use one seed.
- **Open point.** Why the uniform start ($k=0$) is harder than $k=2$ was not resolved.

## Reproduction

Python 3.12, CPU only. From the repository root:

```bash
python -m pip install -r projects/03_neural_operator/requirements.txt
python projects/03_neural_operator/src/run_experiments.py   # about 75 min on 2 cores
python projects/03_neural_operator/src/make_figures.py      # figures and checks
```

[`run_experiments.py`](../src/run_experiments.py) re-runs the Project 1 optimizer to get the designs, computes the FE references, and trains every configuration with a fixed seed (given in each run id). Results are in [`results/`](../results/).

**Checks** (in [`make_figures.py`](../src/make_figures.py), recorded in [`summary.json`](../results/summary.json)):

- The re-run Project 1 design and its compliance match Project 1 exactly, and $\kappa(K)$ matches the Project 2 values.
- For the final PINN fields, $\Pi-\Pi^\star=\tfrac12\mathbf e^{\mathsf T}K\mathbf e$ holds to $10^{-9}$, and the float32 energy matches float64 to $10^{-5}$.
- **Hand check.** The load is $-1$ on the vertical DOF of the top-left node, so compliance equals $-u_y$ there. The Adam PINN gives $u_y=-193.94$ against the FE value $-210.04$, a compliance error of $(210.04-193.94)/210.04=7.66\%$, matching the reported value.

## References

1. MAE 598/494, [Project 3: PINN/PINO in the Field](https://designinformaticslab.github.io/DesignOptimization2025/project3.html) and [AI for Physics: Is Physics Compressible?](https://designinformaticslab.github.io/DesignOptimization2025/ai_for_pde.html)
2. K. Liu, [Project 1 report](../../01_problem_formulation/report/report.md) and [Project 2 report](../../02_gradient_descent/report/report.md).
3. E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, "Efficient topology optimization in MATLAB using 88 lines of code," *Struct. Multidiscip. Optim.*, 43, 1–16, 2011.
4. W. E and B. Yu, "The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems," *Commun. Math. Stat.*, 6, 1–12, 2018.
5. E. Samaniego et al., "An energy approach to the solution of partial differential equations in computational mechanics via machine learning: Concepts, implementation and applications," *Comput. Methods Appl. Mech. Eng.*, 362, 112790, 2020.
6. J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, Chapter 7.
7. A. S. Krishnapriyan, A. Gholami, S. Zhe, R. M. Kirby, and M. W. Mahoney, "Characterizing possible failure modes in physics-informed neural networks," *NeurIPS*, 2021.
8. S. Wang, X. Yu, and P. Perdikaris, "When and why PINNs fail to train: A neural tangent kernel perspective," *J. Comput. Phys.*, 449, 110768, 2022.
9. P. Rathore, W. Lei, Z. Frangella, L. Lu, and M. Udell, "Challenges in training PINNs: A loss landscape perspective," *ICML*, PMLR 235, 2024.
