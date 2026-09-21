# Project 2 — Ill-Conditioned Optimization

**Ill-Conditioning in MBB Beam Analysis**

Kangzheng Liu · OptiForge · MAE 598/494 Design Optimization

The project studies equilibrium as energy minimization for the half MBB beam from Project 1. It examines conditioning under mesh refinement and compares gradient descent, conjugate gradient, and Jacobi preconditioning.

- **[Submission report](report/report.md)**
- [中文译稿](report/report_zh.md)
- [Official assignment](https://designinformaticslab.github.io/DesignOptimization2025/project2.html)
- [Code](src/conditioning_demo.py) · [Numerical results](results/) · [Figures](figures/)

## Reproduce

Use Python 3.12 and run from the repository root:

```bash
python -m pip install -r projects/02_gradient_descent/requirements.txt
python projects/02_gradient_descent/src/conditioning_demo.py
```

The script reads the Project 1 finite-element implementation and saved density field, then writes the Project 2 results and figures. To save a rerun separately, add `--output /tmp/mae598-project2`.
