# Project 4 - PDE-Constrained Design Optimization

- **Weight:** 10%
- **Structure:** compliant gripper from the course menu, a 2D material layout in polypropylene, with large displacements as the nonlinear component (N2)
- **Report website (primary deliverable):** <https://liukz233.github.io/MAE598-Design-Optimization/>, served from [`site/`](site/) by [`.github/workflows/pages.yml`](../../.github/workflows/pages.yml)
- **Same report as Markdown:** [`report/report.md`](report/report.md) (static figures only; the 3D comparison needs the website)
- **Official brief:** [Project 4: PDE-Constrained Design Optimization](https://designinformaticslab.github.io/DesignOptimization2025/project4.html)

## Where each requirement is addressed

| Requirement from the brief | Report section | Code |
|---|---|---|
| Structure, variables, objective, constraints, limits | 1 | [`src/problem.py`](src/problem.py) |
| Own FE analysis; governing PDE, mesh, assumptions | 2 | [`src/mech.py`](src/mech.py), on top of Project 1's [`top88.py`](../01_problem_formulation/src/top88.py) |
| Verification ladder (exact, convergence, balance, invariance, nonlinear, gradients) | 3 | [`src/verify.py`](src/verify.py) |
| Adjoint derivation, gradient checks, Taylor test, cost vs finite differences | 4 | `adjoint_gradient` in [`src/mech.py`](src/mech.py), responses in [`src/problem.py`](src/problem.py) |
| Optimizer, scaling, starts, convergence, active constraints and multipliers | 5 | [`src/mma.py`](src/mma.py), [`src/run_all.py`](src/run_all.py) |
| Nonlinear component, design with and without it | 6 | [`src/run_all.py`](src/run_all.py) |
| Results table, initial vs optimized | 7 | |
| Relation to Projects 1 to 3 | 8 | |
| Bonus: fully nonlinear (N2) | 9 | |
| Limitations and AI-use statement | 10 | |
| 3D comparison (eight viewer requirements) | top of the site | [`site/assets/viewer.js`](site/assets/viewer.js) |
| One command regenerates every number | Reproduction | [`src/run_all.py`](src/run_all.py) |

## Files

| Path | Purpose |
|---|---|
| `src/mech.py` | Finite element model: finite-strain element, Newton solve, stresses, adjoint gradient |
| `src/mma.py` | Method of moving asymptotes |
| `src/problem.py` | Design problem: parameters, supports, loads, objective and constraints |
| `src/verify.py` | Verification ladder V1 to V6 |
| `src/run_all.py` | Runs everything and writes `site/assets/results.json` |
| `src/make_site.py` | Figures, `report/report.md` and `site/index.html` from `results.json` |
| `report/report.template.md` | Report text with placeholders for the numbers |
| `site/` | Published site: `index.html`, `assets/viewer.js`, `assets/results.json`, figures |

`report/report.md` and `site/index.html` are generated. Edit `report/report.template.md` and rerun.

## Reproduction

~~~bash
python -m pip install -r projects/04_engineering_design/requirements.txt
python projects/04_engineering_design/src/run_all.py      # about 5 min; everything
python projects/04_engineering_design/src/make_site.py    # only rebuild figures and pages from results.json
~~~

The scripts read the Project 1 code and results in `projects/01_problem_formulation/`.
