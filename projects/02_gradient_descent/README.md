# Project 2 — Ill-Conditioned Optimization

**Topic:** Ill-conditioning in MBB beam analysis

**Status:** Initial framework and verified pilot experiments; not yet a final submission

**Member:** Kangzheng Liu · OptiForge

Project 2 studies the displacement-equilibrium optimization required by Project 1's SIMP model. It reuses the half-MBB geometry, finite-element implementation, and final density field. The main experiment tests conditioning under mesh refinement; a second experiment tests preconditioning on the saved topology.

- [Working report, mathematical framework, and results](report/report.md)
- [中文思路与评估](report/framework_zh.md)
- [Official assignment](https://designinformaticslab.github.io/DesignOptimization2025/project2.html)
- [Runnable demo](src/conditioning_demo.py)

## Pilot findings

- Uniform meshes from 12 by 4 to 120 by 40: condition number increases from about 15,543 to 1,404,156 and remains large after Jacobi scaling.
- On 12 by 4, GD takes 93,648 updates and CG takes 76 to meet the same physical residual tolerance of $10^{-6}$.
- For the saved topology at $E_{\min}=10^{-9}$, Jacobi scaling lowers the condition number from about $1.19\times10^{12}$ to $1.13\times10^6$. CG exceeds its 5,000-update budget; Jacobi-PCG converges in 1,201 updates.

These are iteration results. They do not establish an end-to-end topology-optimization speedup.

## Assignment coverage

| Diagnostic | Evidence |
|---|---|
| D1: spectrum | Full small-mesh spectra and larger-system spectral endpoints |
| D2: intrinsic test | Mesh growth and survival after diagonal scaling |
| D3: baseline effect | GD histories, common residual tolerance, explicit iteration caps |
| D4: remedy | CG and Jacobi-PCG comparisons; scaled condition numbers |

The report also defines the model, variables, constraints, assumptions, and small-case verification.

## Reproduce

Python 3.12; run from the repository root:

```bash
python -m pip install -r projects/02_gradient_descent/requirements.txt
python projects/02_gradient_descent/src/conditioning_demo.py
```

To rerun without replacing committed outputs, add `--output /tmp/mae598-project2`.

| Directory | Contents |
|---|---|
| `report/` | English working report and Chinese planning analysis |
| `src/` | Reproducible pilot script |
| `figures/` | Spectral, conditioning, and convergence plots |
| `results/` | CSV histories, metrics, checks, and environment metadata |

## Remaining work

- Refine the working report and presentation around the measured mechanisms.
- If adding runtime claims, collect repeated timings with consistent monitoring and separate setup costs.
- Consider stronger SPD preconditioning only as an optional extension.

The existing directory name is retained so repository links continue to work.
