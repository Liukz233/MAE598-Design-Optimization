# Project 2 — Ill-Conditioned Optimization

![Project 2 cover: optimized beam and equilibrium convergence](figures/project2_cover.png)

**Solving the MBB Beam Equilibrium Problem**

Kangzheng Liu · OptiForge · MAE 598/494 Design Optimization

Project 1 optimized the beam's material layout. Project 2 studies the equilibrium solve inside that workflow. Gradient descent slows as the mesh is refined, and its iteration count follows from the softest deformation mode. CG handles the mesh part of the ill-conditioning; Jacobi preconditioning removes the solid–void contrast of the optimized layout. The report checks the result against the Project 1 displacement solution.

- **[Submission report](report/report.md)**
- [Official assignment](https://designinformaticslab.github.io/DesignOptimization2025/project2.html)
- [Code](src/conditioning_demo.py) · [Numerical results](results/) · [Figures](figures/)

## Reproduce

Use Python 3.12 and run from the repository root:

```bash
python -m pip install -r projects/02_gradient_descent/requirements.txt
python projects/02_gradient_descent/src/conditioning_demo.py
```

The script reads the Project 1 finite-element implementation and saved density field, then writes the Project 2 results and figures. The run takes about 2.5 minutes, mostly for gradient descent. To save a rerun separately, add `--output /tmp/mae598-project2`.
