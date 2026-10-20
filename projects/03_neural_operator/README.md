# Project 3 - PINN/PINO in the Field

- **Weight:** 10%
- **Primary report:** [`report/report.md`](report/report.md)
- **Official brief:** [Project 3: PINN/PINO in the Field](https://designinformaticslab.github.io/DesignOptimization2025/project3.html)
- **Researcher:** Dr. Leixin Ma (OASIS Lab, ASU)
- **Path:** C (diagnosis and fix)

## Requirements map

| Requirement from the official brief | Where addressed in the report |
|---|---|
| Outreach log | Section 1 |
| Problem brief (researcher-checked) | Section 2 |
| Minimal case: PDE, domain, BCs, parameters, difficulty knob, what was simplified, path | Section 3 |
| P1 baseline (conventional solver, accuracy, wall-clock) | Section 4.1 |
| P2 loss components logged separately, log scale | Section 4.3 |
| P3 relative L2 error vs. knob | Section 4.2 |
| P4 error field and spectrum | Section 4.4 |
| P5 training/inference time, hardware, break-even $N^\star$ | Section 4.6 |
| Path B: failure along the knob + controlled experiment separating two causes | Sections 4.2 and 4.4 |
| Path C: remedy targeting the diagnosed cause, before/after with P2–P5 | Section 4.5 |
| One-page recommendation memo, sent to the researcher | Section 5 |
| Assumptions and limitations | Section 6 |
| Runnable code, fixed seeds, instructions | Reproduction |

## Workspace

| Path | Purpose |
|---|---|
| `report/report.md` | Public report and grading entry point |
| `src/pinn_core.py` | FE energy, network, training loop, OC snapshots |
| `src/run_experiments.py` | Runs every configuration in the report |
| `src/make_figures.py` | Figures, summary numbers, and checks |
| `figures/` | Report figures |
| `results/` | Run table, training histories, final fields, reference solves |

## Reproduction

From the repository root, with Python 3.12:

~~~bash
python -m pip install -r projects/03_neural_operator/requirements.txt
python projects/03_neural_operator/src/run_experiments.py   # about 75 min on 2 CPU cores
python projects/03_neural_operator/src/make_figures.py      # about 30 s
~~~
