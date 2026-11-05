---
layout: default
title: "Compliant gripper: PDE-constrained design optimization"
---

# Compliant gripper: PDE-constrained design optimization

MAE 598 Design Optimization, Project 4 · Team OptiForge (Kangzheng Liu)

**Status:** structure chosen; report in progress. The final report goes on this page by Nov 25, 2026.

## Structure

A compliant gripper is one piece of material with no pins or sliding joints. A push at the input deforms its thin flexures, and the deformation closes the jaws. The design variable is the material layout in the plane: one density per finite element (SIMP), extruded for display.

## Planned formulation

- **Objective:** jaw closing displacement under the input force, with a spring at the jaws that stands in for the gripped object.
- **Constraints:** material volume, peak strain (aggregated), and an input-side limit. Formulas and limits will be listed here with their sources.
- **Physics:** 2D elasticity on four-node elements. The nonlinear component is N2: geometrically nonlinear analysis solved by Newton's method. Gradients come from one adjoint solve with the tangent stiffness at the converged state:

$$
\mathbf R(\mathbf u;\boldsymbol\rho)=\mathbf 0,\qquad
\mathbf K_T^{\mathsf T}\boldsymbol\lambda=-\left(\frac{\partial f}{\partial \mathbf u}\right)^{\mathsf T},\qquad
\frac{\mathrm d f}{\mathrm d\boldsymbol\rho}=\frac{\partial f}{\partial\boldsymbol\rho}+\boldsymbol\lambda^{\mathsf T}\frac{\partial\mathbf R}{\partial\boldsymbol\rho}.
$$

## Design domain (preliminary)

<div id="setup-viewer" class="viewer"></div>
<p class="caption">Drag to orbit. Blue blocks: design domain, extruded to its thickness. Dark pyramids: clamped supports. Orange arrow: input force. Blue arrows: jaw closing direction at the output points. The gap between the jaws is kept void. Dimensions are placeholders until the formulation is fixed.</p>
<script type="module" src="assets/setup-viewer.js"></script>
