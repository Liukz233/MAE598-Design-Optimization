# Project 4: Lightest Heat-Spreader Plate for a Sealed Electronics Enclosure

**Kangzheng Liu · OptiForge · MAE 598/494 Design Optimization**  
**Official brief:** [Project 4: PDE-Constrained Design Optimization](https://designinformaticslab.github.io/DesignOptimization2025/project4.html)  
**Structure:** heat sink for a power chip (from the menu) · **Nonlinear component:** N2, surface radiation · **Code:** [repository](https://github.com/Liukz233/MAE598-Design-Optimization/tree/main/projects/04_engineering_design)

## Summary

A sealed, fanless electronics enclosure has no vents, so its aluminium lid is also its heat sink. The lid here is a [[config.Lx_mm|.0f]] × [[config.Ly_mm|.0f]] mm plate with a [[components.chip.power_W|.0f]] W power module and a [[components.optic.power_W|.1f]] W optical transceiver mounted on the inside. The outside loses heat to [[config.T_inf_C|.0f]] °C air by natural convection and radiation. I chose the thickness of each [[config.tile_mm|.1f]] mm tile of the plate ([[config.n_variables]] variables, [[config.t_min_mm|.0f]] to [[config.t_max_mm|.0f]] mm) to minimize mass while keeping the module at or below [[components.chip.limit_C|.0f]] °C and the transceiver at or below [[components.optic.limit_C|.0f]] °C.

- The optimized plate weighs [[designs.optimized.metrics.mass_g|.1f]] g and meets both limits. The uniform 2 mm starting plate weighs [[designs.initial.metrics.mass_g|.0f]] g and violates both, with the module at [[designs.initial.metrics.T_chip_C|.1f]] °C.
- No uniform plate is feasible. Thickening the plate cools the module but warms the transceiver. The thinnest uniform plate that satisfies the module constraint is [[uniform_chip_feasible.thickness_mm|.2f]] mm ([[uniform_chip_feasible.mass_g|.0f]] g), and it leaves the transceiver at [[uniform_chip_feasible.T_optic_C|.1f]] °C.
- Radiation carries [[designs.optimized.metrics.radiation_percent|.0f]]% of the heat, so the state equation is nonlinear in temperature. It is solved by Newton's method in [[designs.optimized.metrics.newton_iterations]] iterations, and the gradients of both constraints come from the adjoint at the converged state in [[gradient_cost.adjoint_seconds|.3f]] s, against [[gradient_cost.fd_seconds|.0f]] s for finite differences.
- The nonlinearity changes the design. If radiation is linearized about ambient, the optimizer needs [[models.linearized.mass_g|.0f]] g, [[nonlinear.extra_mass_linearized_percent|.0f]]% more. Checked with the nonlinear model, that plate is [[models.linearized.nonlinear_model.margin_chip_K|.1f]] K and [[models.linearized.nonlinear_model.margin_optic_K|.1f]] K below the two limits, so the extra metal is not needed.
- The solver passes all [[verification.n_checks]] checks of the verification ladder (Section 3).

<!--VIEWER-->

<p class="caption">Left: the uniform starting plate. Right: the optimized plate. Drag to rotate (the two views share one camera) and scroll to zoom. Plan dimensions are to scale; thickness is drawn 5 times too large by default, and the slider changes that factor. The flat mounting face is at the bottom. Colour is the plate temperature from the nodal values of the finite element solution; each element is drawn as four triangles around its centre value, which is what the bilinear shape functions give there. The play button steps the right panel through the optimizer's iterations.</p>

## 1. Structure and formulation

**Why it matters.** Outdoor radio units, LED drivers and similar equipment are sealed against dust and water, so they cannot use fans. All heat leaves through the enclosure walls, and the wall thickness sets how well heat spreads from a hot component to the rest of the surface. A thicker wall runs the hot component cooler but adds weight and, as shown below, warms the neighbouring parts.

**Setup.** The plate is machined from [[config.t_max_mm|.0f]] mm stock, with pockets cut from the outer face. Components sit on the flat inner face.

| Quantity | Value | Source |
|---|---|---|
| Plate | [[config.Lx_mm|.0f]] × [[config.Ly_mm|.0f]] mm, aluminium 6061-T6, $k=[[config.k|.0f]]$ W/(m·K), $\rho=[[config.rho|.0f]]$ kg/m³ | Data sheet [3] |
| Surface | Anodized, emissivity $\varepsilon=[[config.eps|.2f]]$ | [2], Table A.11 |
| Convection | $h=[[config.h|.0f]]$ W/(m²·K), outer face only | Free convection in gases is 2 to 25 W/(m²·K) [2, Table 1.1]; plate correlations at a 30 K rise give 5 to 6 [2, Ch. 9] |
| Ambient and surroundings | $T_\infty=[[config.T_inf_C|.0f]]$ °C | Hot-day design assumption |
| Power module | [[components.chip.power_W|.0f]] W on 25 × 25 mm, mounting face at most [[components.chip.limit_C|.0f]] °C | Assumed device: junction limit 125 °C, less 20 W × 1.5 K/W from junction to plate, less a 10 K margin |
| Optical transceiver | [[components.optic.power_W|.1f]] W on 25 × 25 mm, plate under it at most [[components.optic.limit_C|.0f]] °C | Commercial-grade case rating 70 °C at 1.5 W [4], less 1.5 W × 4 K/W for the cage and pad (assumed), less a 2 K margin |
| Thickness | [[config.t_min_mm|.0f]] to [[config.t_max_mm|.0f]] mm | Stock plate; 1 mm is the thinnest floor I assume can be machined |

**Design variables.** One variable per tile, $x_j\in[0,1]$, $j=1,\dots,[[config.n_variables]]$, with tile thickness

```math
t_j = t_{\min} + (t_{\max}-t_{\min})\,x_j ,\qquad t_{\min}=1\ \text{mm},\quad t_{\max}=8\ \text{mm}.
```

**Objective and constraints.** With $\theta=T-T_\infty$ the temperature rise, $A_{\text{tile}}$ the tile area, and $\mathcal N_c$ the [[config.ks_nodes.chip]] mesh nodes under component $c$:

```math
\begin{aligned}
\min_{\mathbf x}\quad & m(\mathbf x)=\rho A_{\text{tile}}\sum_j t_j \\
\text{s.t.}\quad & g_c(\mathbf x)=\mathrm{KS}\!\left(\frac{\theta_i}{\theta_c^{\max}},\ i\in\mathcal N_c\right)-1\le 0,\qquad c\in\{\text{module},\ \text{transceiver}\},\\
& 0\le x_j\le 1 ,
\end{aligned}
```

where $\theta^{\max}_{\text{module}}=45$ K, $\theta^{\max}_{\text{transceiver}}=22$ K, and $T$ solves the state equation of Section 2. The peak temperature under a component is not differentiable where the hottest node changes, so each constraint uses the Kreisselmeier–Steinhauser aggregate [5]

```math
\mathrm{KS}(\mathbf a)=a_{\max}+\frac1{\rho_{KS}}\ln\sum_i e^{\rho_{KS}(a_i-a_{\max})},\qquad
a_{\max}\le \mathrm{KS}(\mathbf a)\le a_{\max}+\frac{\ln N}{\rho_{KS}} .
```

With $\rho_{KS}=[[config.ks_rho|.0f]]$ and $N=[[config.ks_nodes.chip]]$ the aggregate overestimates the peak by at most [[config.ks_bound_K.chip|.2f]] K for the module and [[config.ks_bound_K.optic|.2f]] K for the transceiver. I report the true peaks next to the aggregate.

## 2. Analysis model

**Governing equation.** The plate is thin and conducts well, so its temperature is uniform through the thickness: the Biot number $(h+h_r)\,t/k$ is [[config.biot|.4f]] at the thickest point. An energy balance on a column of plate gives the fin equation in two dimensions,

```math
-\nabla\!\cdot\!\big(k\,t\,\nabla T\big)+h\,(T-T_\infty)+\varepsilon\sigma\,(T^4-T_\infty^4)=q
\ \ \text{in }\Omega,\qquad k\,t\,\frac{\partial T}{\partial n}=0\ \ \text{on }\partial\Omega ,
```

with $T$ in kelvin, $t(x,y)$ the thickness, and $q$ the heat flux from the components ($P_c/A_c$ under each one, zero elsewhere). Conduction is in the plane, and the surface terms act on the outer face. The $T^4$ term makes the equation nonlinear.

**Discretization.** Four-node bilinear rectangles on a [[config.nx]] × [[config.ny]] grid ([[config.element_mm|.2f]] mm elements, [[config.n_nodes]] unknowns). Each design tile covers 2 × 2 elements. The discrete residual is

```math
\begin{aligned}
&\mathbf R(\mathbf T;\mathbf t)=\mathbf K_c(\mathbf t)\,\mathbf T+h\,\mathbf M\,(\mathbf T-T_\infty\mathbf 1)+\mathbf r(\mathbf T)-\mathbf Q=\mathbf 0,\\
&\mathbf K_c=\sum_e k\,t_e\,\mathbf k_0,\qquad
r_i=\varepsilon\sigma\!\int_\Omega N_i\,(T_h^4-T_\infty^4)\,dA ,
\end{aligned}
```

where $\mathbf k_0=\int\nabla N^{\mathsf T}\nabla N\,dA$ is the element conduction matrix for unit $k\,t$, $\mathbf M=\int N^{\mathsf T}N\,dA$, and $T_h=\sum_i N_iT_i$. The radiation integrals use 3 × 3 Gauss points, which is exact for these polynomials. Newton's method starts from $T_\infty$ and uses the tangent

```math
\mathbf K_T=\frac{\partial\mathbf R}{\partial\mathbf T}=\mathbf K_c+h\,\mathbf M+\varepsilon\sigma\!\int_\Omega 4\,T_h^3\,N^{\mathsf T}N\,dA ,
```

which is symmetric positive definite. It stops when $\lVert\mathbf R\rVert/\lVert\mathbf Q\rVert<10^{-10}$. Global matrices are sparse (`scipy.sparse`), and the solver is my own code in [`plate.py`](https://github.com/Liukz233/MAE598-Design-Optimization/blob/main/projects/04_engineering_design/src/plate.py); no FE library is used.

**Assumptions.**

- Temperature is uniform through the thickness (small Biot number). Spreading through the thickness directly under the module is ignored.
- $h$ is constant. In natural convection it grows roughly as $\theta^{1/4}$.
- The surface is gray and diffuse and sees large surroundings at $T_\infty$.
- Only the outer face exchanges heat. The inner face faces the electronics and is taken as adiabatic, as are the plate edges.
- Steps between tiles do not add surface area.
- Each component is a uniform heat flux on its footprint. Interface resistances are accounted for in the limits, not in the model.

**Mesh check.** The optimized design re-analysed on coarser and finer meshes:

[[TABLE:mesh]]

The peaks change by less than 0.1 K, which is smaller than the KS margin.

## 3. Verification

All checks run from [`verify.py`](https://github.com/Liukz233/MAE598-Design-Optimization/blob/main/projects/04_engineering_design/src/verify.py) each time the results are regenerated.

[[TABLE:ladder]]

For the two invariance rows the exact value is zero, and the error is the difference divided by the largest temperature rise.

Where the exact answers come from:

**V1, linear fin.** A strip of constant thickness with heat flow $q'$ per unit width entering at $x=0$, an insulated tip, and convection only obeys $k\,t\,\theta''=h\,\theta$. With $m=\sqrt{h/(k\,t)}$, the solution is $\theta(x)=q'\cosh\!\big(m(L-x)\big)/\big(k\,t\,m\sinh mL\big)$ [2, Ch. 3].

**V2, manufactured solution.** I picked $\theta^\star=\theta_0\cos(\pi x/L_x)\cos(\pi y/L_y)$, which has zero flux on all edges, and a thickness $t=t_0(1+x/L_x)$. Substituting into the full equation, including the $T^4$ term, gives the source $q$ that makes $\theta^\star$ exact. The $L_2$ error of bilinear elements should fall as the square of the element size; the measured slope is [[verification.mms.slope|.2f]].

**V3, heat balance.** The shape functions sum to one, so summing all rows of $\mathbf R=\mathbf 0$ gives heat out equal to heat in.

**V4, zero modes and invariance.** A uniform temperature conducts no heat, so $\mathbf K_c$ must have exactly one zero eigenvalue. This is the thermal counterpart of a rigid-body mode. Mirroring or rotating the plate together with its thickness field and loads must return the same temperatures.

**V5, radiation.** (a) A plate heated uniformly with radiation only stays uniform at $T=(T_\infty^4+q/\varepsilon\sigma)^{1/4}$. (b) For a fin with radiation only, multiplying $k\,t\,T''=\varepsilon\sigma(T^4-T_\infty^4)$ by $T'$ and integrating from the base to the insulated tip gives an exact relation between the base heat flow and the two end temperatures,

```math
q'^2=2\,k\,t\,\varepsilon\sigma\left[\frac{T_0^5-T_L^5}{5}-T_\infty^4\,(T_0-T_L)\right].
```

I apply $q'$, read $T_0$ and $T_L$ from the solver, and compare the right-hand side with the applied value. (c) Newton's residuals on the design model are [[verification.newton.residuals.1|.1e]], [[verification.newton.residuals.2|.1e]], [[verification.newton.residuals.3|.1e]], [[verification.newton.residuals.4|.1e]], an observed order of [[verification.newton.order|.2f]].

**V6, gradients.** See Section 4.

![Verification plots](assets/fig_verification.png)

## 4. Sensitivities

**Adjoint.** A constraint depends on the design only through the temperature, $g=g(\mathbf T(\mathbf t))$, and $\mathbf T$ is defined implicitly by $\mathbf R(\mathbf T;\mathbf t)=\mathbf 0$. Differentiating the residual gives $\mathbf K_T\,d\mathbf T/dt_e=-\partial\mathbf R/\partial t_e$. Introducing the adjoint $\boldsymbol\lambda$ removes the need to solve this for every element:

```math
\mathbf K_T^{\mathsf T}\boldsymbol\lambda=-\left(\frac{\partial g}{\partial\mathbf T}\right)^{\mathsf T},
\qquad
\frac{dg}{dt_e}=\boldsymbol\lambda^{\mathsf T}\frac{\partial\mathbf R}{\partial t_e}
=k\,\boldsymbol\lambda_e^{\mathsf T}\mathbf k_0\mathbf T_e .
```

Only the conduction matrix depends on the thickness, so $\partial\mathbf R/\partial t_e=k\,\mathbf k_0\mathbf T_e$ involves a single element. The chain rule to the design variables is $dg/dx_j=(t_{\max}-t_{\min})\sum_{e\in\text{tile }j}dg/dt_e$. For the KS constraint, $\partial g/\partial T_i=w_i/\theta^{\max}_c$ with $w_i=e^{\rho_{KS}a_i}/\sum_l e^{\rho_{KS}a_l}$ on the component's nodes and zero elsewhere.

The nonlinear solve does not need to be differentiated step by step. By the implicit function theorem only $\mathbf K_T$ at the converged temperature matters. $\mathbf K_T$ is symmetric and its factorization from the last Newton step is reused, so each constraint gradient costs one back-substitution. $\boldsymbol\lambda$ is the Lagrange multiplier of the state equation in the KKT conditions of the full problem. The mass is explicit: $\partial m/\partial x_j=\rho A_{\text{tile}}(t_{\max}-t_{\min})$. The linearized model of Section 6 uses the same formulas with a constant $\mathbf K_T$.

**Gradient check.** Directional derivatives $\nabla g\cdot\mathbf v$ from the adjoint against central differences, at a random design, for three random directions per constraint (worst case shown). The design has one kind of variable, tile thickness.

[[TABLE:gradients]]

In the Taylor test (right panel of the figure in Section 3) the central-difference error falls as $\epsilon^2$ (fitted slope [[verification.taylor.slope|.2f]]) down to about $10^{-8}$, then rises again from round-off. The largest error in the table belongs to a direction along which the derivative is small, so round-off in the difference is relatively larger.

**Cost.** For the [[gradient_cost.n_variables]] variables, one adjoint evaluation (Newton solve plus both constraint gradients) takes [[gradient_cost.adjoint_seconds|.3f]] s. Forward differences need [[gradient_cost.fd_solves]] Newton solves and take [[gradient_cost.fd_seconds|.1f]] s, [[gradient_cost.speedup|.0f]] times longer. The two gradients agree to [[gradient_cost.max_difference|.0e]] of the largest component.

## 5. Optimization

**Setup.** SLSQP from SciPy [6] with the adjoint gradients. The variables are already in $[0,1]$. The objective is the mass divided by the mass of the full 8 mm plate ([[config.mass_full_plate_g|.0f]] g), and the constraints are written as $g_c\le0$ with $g_c$ of order one. The start is the uniform 2 mm plate.

**Convergence.** [[optimization.iterations]] iterations and [[optimization.seconds|.0f]] s. Both constraints start violated and are satisfied within the first few iterations; the rest of the run removes mass.

![Optimization history](assets/fig_history.png)

**Other starts.** Four starts reach the same design:

[[TABLE:starts]]

**Active constraints and multipliers.** Both temperature constraints are active. At the optimum [[optimization.kkt.n_lower]] tiles sit at 1 mm, [[optimization.kkt.n_upper]] at 8 mm, and [[optimization.kkt.n_free]] are in between. With SLSQP's multipliers $\mu_{\text{module}}=[[optimization.kkt.multipliers.chip|.3f]]$ and $\mu_{\text{transceiver}}=[[optimization.kkt.multipliers.optic|.3f]]$, the stationarity residual $\nabla\hat m+\sum_c\mu_c\nabla g_c$ on the free tiles is at most [[optimization.kkt.stationarity_free_percent|.2f]]% of $\partial\hat m/\partial x_j$. On the tiles at the lower bound it is positive and on those at the upper bound it is negative, as the KKT conditions require.

A multiplier is the rate at which the optimal mass falls when its limit is relaxed. Since $g_c=\theta/\theta_c^{\max}-1$,

```math
\frac{dm^\star}{dT_c^{\max}}=-\frac{\mu_c\,m_{\text{full}}}{\theta_c^{\max}} .
```

- **Power module:** [[optimization.kkt.saving_chip_g_per_K|.1f]] g per kelvin. Re-optimizing with the limit raised by [[optimization.kkt.relaxation_K|.1f]] K gives [[optimization.kkt.reoptimized_saving_chip_g_per_K|.1f]] g/K.
- **Transceiver:** [[optimization.kkt.saving_optic_g_per_K|.1f]] g per kelvin (re-optimized: [[optimization.kkt.reoptimized_saving_optic_g_per_K|.1f]] g/K).

In engineering terms, each kelvin of margin given up at the module is worth [[optimization.kkt.saving_chip_percent_per_K|.1f]]% of the plate's mass. The transceiver limit is active but cheap: removing it altogether saves only [[optimization.kkt.cost_of_optic_limit_g|.1f]] g, and the transceiver would then sit at [[optimization.kkt.T_optic_without_limit_C|.1f]] °C.

**What the optimizer did.** It left most of the plate at 1 mm and built a mound under the module that tapers outwards, which spreads the 20 W over enough surface to bring the module down to its limit. The mound reaches further towards the middle of the plate, where there is more surface to feed, than towards the near edge. It stops well short of the transceiver, which gets a small pad of its own to spread its 1.5 W.

**Why no uniform plate works.** A thicker plate spreads the module's heat further and flattens the temperature field towards its mean of about [[designs.optimized.metrics.T_mean_C|.0f]] °C. That is below the module limit but above the transceiver limit, so the transceiver gets warmer as the plate gets thicker.

## 6. Nonlinear component (N2)

The nonlinear part of the model is the radiation term $\varepsilon\sigma(T^4-T_\infty^4)$ in the state equation. It carries [[designs.optimized.metrics.heat_radiation_W|.1f]] of the [[config.power_W|.1f]] W. To see what the nonlinearity does to the design, I repeated the optimization with the usual linearization about ambient,

```math
\varepsilon\sigma\,(T^4-T_\infty^4)\approx h_r\,(T-T_\infty),\qquad h_r=4\,\varepsilon\sigma T_\infty^3=[[nonlinear.h_r_linearized|.2f]]\ \text{W/(m}^2\text{K)},
```

which makes the state equation linear, and then checked that design with the nonlinear model.

[[TABLE:models]]

![Optimized plates for the two radiation models](assets/fig_designs.png)

The linearized model asks for [[nonlinear.extra_mass_linearized_g|.0f]] g more aluminium ([[nonlinear.extra_mass_linearized_percent|.0f]]%) and a different layout: a wider mound at the module, a 1 mm gap between the module and the transceiver, and a second mound at the transceiver that extends towards the far edge. In the nonlinear model that plate is cooler than it needs to be at both components.

The reason is that the true radiation coefficient $\varepsilon\sigma(T^2+T_\infty^2)(T+T_\infty)$ grows with temperature. On the optimized plate it is [[nonlinear.h_r_at_coldest|.2f]] W/(m²·K) at the coldest point and [[nonlinear.h_r_at_chip|.2f]] at the module, against [[nonlinear.h_r_linearized|.2f]] in the linearization. The linear model therefore underestimates the heat leaving everywhere, and most where the plate is hottest. A linear model also fixes the mean plate temperature: integrating the equation over the plate gives $\bar\theta=P/\big((h+h_r)A\big)$ for every thickness field, [[nonlinear.mean_T_linearized_C|.1f]] °C here, where the nonlinear solution has [[designs.optimized.metrics.T_mean_C|.1f]] °C. The linearized optimizer has to hold the transceiver that much further below the mean, and it can do so only by isolating the transceiver from the module and giving it its own fin area.

Without radiation at all the problem has no solution. Convection alone fixes the mean plate temperature at $T_\infty+P/(hA)=[[nonlinear.mean_T_convection_only_C|.1f]]$ °C for every design, above both limits. The full 8 mm plate has the module at [[nonlinear.convection_only_thickest_plate.T_chip_C|.1f]] °C.

## 7. Results

All values are from the nonlinear model.

[[TABLE:results]]

The optimized plate is [[optimization.saving_vs_initial_percent|.0f]]% lighter than the starting plate and [[optimization.saving_vs_uniform_chip_percent|.0f]]% lighter than the uniform plate that satisfies the module constraint alone. Its true peaks are [[designs.optimized.metrics.margin_chip_K|.2f]] K and [[designs.optimized.metrics.margin_optic_K|.2f]] K below the limits; that small margin is the KS overestimate.

## 8. Bonus

**Fully nonlinear (N2).** The nonlinear component is a nonlinear state equation solved by Newton's method, with gradients from implicit differentiation at the converged state. It is checked against two radiation benchmarks with closed-form answers (V5 a and b), the Newton convergence order (V5 c), and finite differences (V6).

## 9. Limitations and AI use

**Limitations.**

- The convection coefficient is constant and the same over the whole face. A correlation $h\propto\theta^{1/4}$ would add a second nonlinearity of the same kind and could be handled by the same Newton solver and adjoint.
- The plate has no fins. The design question here is how to spread heat in a wall of fixed area, not how to add area.
- Thickness changes in steps between tiles, with no limit on the step size.
- The thin-plate model ignores spreading through the thickness under the module, where the plate is 8 mm thick and the module is 25 mm wide. I did not check this against a 3D model.
- One steady load case at the hottest ambient. Material properties are constants.
- The module and transceiver limits rest on assumed device data (Section 1).

**AI use.** I worked with a coding agent (Claude). It proposed the structure and the numbers in Section 1, wrote the solver, the adjoint, the verification and optimization scripts, and the three.js viewer, and drafted the text of this page. The exact answers in the ladder are the closed forms written out in Section 3, so that they can be checked by hand independently of the code. Every row of that table is recomputed whenever the results are regenerated.

One bug the checks caught: in the first uniform-plate runs for the mesh check, Newton's method stalled on the 8 mm plate at a relative residual of $1.5\times10^{-11}$, just above the tolerance of $10^{-11}$, and reported failure. The linear terms of the residual were being applied to the absolute temperature (about 340 K) instead of the rise, so round-off in $\mathbf K_c\mathbf T$ set a floor on the residual. The residual now uses $\theta=T-T_\infty$, which is exact because $\mathbf K_c\mathbf 1=\mathbf 0$ (check V4), and the tolerance is $10^{-10}$. Newton's last residual is now about [[verification.newton.residuals.4|.0e]].

## Reproduction

Python [[meta.python]], NumPy [[meta.numpy]], SciPy [[meta.scipy]]. From the repository root:

```bash
python -m pip install -r projects/04_engineering_design/requirements.txt
python projects/04_engineering_design/src/run_all.py     # about [[meta.run_minutes|.0f]] min
```

This one command runs the verification ladder and all optimizations, writes [`site/assets/results.json`](https://github.com/Liukz233/MAE598-Design-Optimization/blob/main/projects/04_engineering_design/site/assets/results.json), draws the figures, and rebuilds this page from [`report/report.template.md`](https://github.com/Liukz233/MAE598-Design-Optimization/blob/main/projects/04_engineering_design/report/report.template.md). Every number in the text, the tables and the 3D viewer is read from `results.json`.

## References

1. MAE 598/494, [Project 4: PDE-Constrained Design Optimization](https://designinformaticslab.github.io/DesignOptimization2025/project4.html).
2. T. L. Bergman, A. S. Lavine, F. P. Incropera, and D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley, 2011.
3. ASM Aerospace Specification Metals, [Aluminum 6061-T6 data sheet](https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=MA6061T6).
4. NVIDIA, [MMA2L20-AR 25GbE SFP28 LR transceiver specifications](https://docs.nvidia.com/networking/display/MMA2L20ARSpec/Specifications): operating case temperature 0 to 70 °C, power consumption at most 1.5 W.
5. G. Kreisselmeier and R. Steinhauser, "Systematic control design by optimizing a vector performance index," *IFAC Proceedings Volumes*, 12(7), 313–318, 1979.
6. D. Kraft, "A software package for sequential quadratic programming," DFVLR-FB 88-28, 1988; implemented in `scipy.optimize.minimize(method="SLSQP")`.
