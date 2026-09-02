# Project 1 - Optimization Problem Formulation

- **Weight:** 5%
- **Status:** Formulation and computational result completed; ready for review
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

| Official requirement | Points | Template location |
|---|---:|---|
| Problem identification and motivation | 15 | [Section 1](report/report.md#1-problem-identification-and-motivation) |
| Decision variables, units, dimensions, bounds, and types | 15 | [Section 2](report/report.md#2-decision-variables) |
| Explicit objective function and min/max direction | 25 | [Section 3](report/report.md#3-objective-function) |
| Complete equality and inequality constraints with meaning | 25 | [Section 4](report/report.md#4-constraints) |
| Classification and structural justification | 15 | [Section 5](report/report.md#5-problem-classification) |
| Clear, professional GitHub Markdown | 5 | Entire report |
| Assumptions and omitted real-world effects | Required | [Section 6](report/report.md#6-assumptions-and-simplifications) |

## Optional computational bonus

| Bonus item | Points | Template location |
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
pip install -r requirements.txt
python projects/01_problem_formulation/src/top88.py
~~~

| Output | Link |
|---|---|
| Final topology | [PNG](figures/final_topology.png) |
| Convergence plots | [PNG](figures/convergence.png) |
| Iteration history | [CSV](results/history.csv) |
| Final density field | [CSV](results/final_density.csv) |
| Run summary | [JSON](results/summary.json) |

## Project-specific checklist

- [ ] A real engineering stakeholder and decision need are identified
- [ ] Density design variables are distinguished from FEM state variables
- [ ] Every symbol in the objective and constraints is defined
- [ ] Equality constraints, inequality constraints, and bounds are separated
- [ ] Nonconvexity is justified from the SIMP/FEM structure
- [ ] Assumptions and uncaptured effects are stated
- [ ] All bracketed prompts are removed from the final report
- [ ] Bonus code runs from a clean environment, if included
- [ ] Report math, figures, citations, and relative links render on GitHub
- [ ] Repository and report URLs open while signed out
