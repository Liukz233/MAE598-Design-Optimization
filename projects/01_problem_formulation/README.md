# Project 1 - Optimization Problem Formulation

- **Weight:** 5%
- **Status:** Final report and computational verification completed
- **Selected topic:** SIMP topology optimization of the symmetric half MBB beam
- **Official brief:** [Project 1: Optimization Problem Formulation](https://designinformaticslab.github.io/DesignOptimization2025/project1_optimization_formulation.html)
- **Primary report:** [`report/report.md`](report/report.md)

## Required submission format

The official brief requires:

1. One Markdown report in the team's collaborative GitHub repository.
2. A publicly accessible repository and report.
3. Submission of the repository URL through Canvas.
4. GitHub-rendered mathematics, with figures and cited sources where helpful.

The Markdown report is the grading artifact. Code and computational results are optional for the base score but can earn bonus credit.

## Required report sections

| Official requirement | Points | Report location |
|---|---:|---|
| Problem identification and motivation | 15 | [Section 1](report/report.md#1-problem-identification-and-motivation) |
| Decision variables, units, dimensions, bounds, and types | 15 | [Section 2](report/report.md#2-decision-variables-and-fixed-inputs) |
| Explicit objective function and min/max direction | 25 | [Section 3](report/report.md#3-objective-and-complete-optimization-formulation) |
| Complete equality and inequality constraints with meaning | 25 | [Section 4](report/report.md#4-physical-meaning-of-the-objective-and-constraints) |
| Classification and structural justification | 15 | [Section 5](report/report.md#5-problem-classification-and-source-of-nonconvexity) |
| Clear, professional GitHub Markdown | 5 | Entire report |
| Assumptions and omitted real-world effects | Required | [Section 6](report/report.md#6-assumptions-and-scope) |

## Optional computational bonus

| Bonus item | Points | Report location |
|---|---:|---|
| Solution methodology | 8 | Section 7.1 |
| Results and interpretation | 8 | Section 7.2 |
| Code and reproducibility | 4 | Section 7.3 |

## Workspace

| Path | Purpose |
|---|---|
| `report/report.md` | Rubric-aligned public report |
| `notebooks/` | Exploratory SIMP/OC implementation and checks |
| `src/` | Reusable FEM, sensitivity, filtering, and OC code |
| `figures/` | Problem schematic, convergence history, and final topology |
| `results/` | Small numerical summaries and parameter files |

## Reproduce the submitted result

The calculation reproduces the official DTU example
`top88(120, 40, 0.5, 3.0, 3.5, 1)` with a readable Python implementation.

~~~bash
pip install -r projects/01_problem_formulation/requirements.txt
python projects/01_problem_formulation/src/top88.py
python projects/01_problem_formulation/src/verify_top88.py
~~~

| Output | Link |
|---|---|
| Final topology | [PNG](figures/final_topology.png) |
| Convergence plots | [PNG](figures/convergence.png) |
| Iteration history | [CSV](results/history.csv) |
| Final density field | [CSV](results/final_density.csv) |
| Run summary | [JSON](results/summary.json) |
| Numerical verification | [JSON](results/verification.json) |

## Project-specific checklist

- [x] A real engineering stakeholder and decision need are identified
- [x] Density design variables are distinguished from FEM state variables
- [x] Every symbol in the objective and constraints is defined
- [x] Equality constraints, inequality constraints, and bounds are separated
- [x] Nonconvexity is justified from the SIMP/FEM structure
- [x] Assumptions and uncaptured effects are stated
- [x] All bracketed prompts are removed from the final report
- [x] Bonus code reruns successfully with the listed scientific Python dependencies
- [x] Report math, figures, citations, and relative links render on GitHub
- [x] Repository and report URLs open while signed out

## Submission entry

Submit the [repository URL](https://github.com/Liukz233/MAE598-Design-Optimization) in Canvas, as specified by the brief. The root README prominently links to the [Project 1 report](report/report.md). Repository readiness does not record Canvas submission or completion of any assigned rehearsal or presentation.
