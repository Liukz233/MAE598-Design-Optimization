<p align="center">
  <img src="assets/optiforge-logo.png" alt="OptiForge logo" width="180">
</p>

<h1 align="center">OptiForge</h1>

<p align="center">
  Course projects for <strong>MAE 598/494 Design Optimization</strong>, Fall 2026
</p>

## Team

- **Team:** OptiForge
- **Member:** Kangzheng Liu
- **Format:** Individual team

OptiForge reflects the process of turning an initial design into a refined solution through modeling, analysis, and iterative optimization.

## Project index

| Project | Topic | Weight | Workspace | Status |
|---:|---|---:|---|---|
| 1 | Problem formulation | 5% | [Open project](projects/01_problem_formulation/) | Pending details |
| 2 | Gradient descent | 5% | [Open project](projects/02_gradient_descent/) | Pending details |
| 3 | Neural operator | 10% | [Open project](projects/03_neural_operator/) | Pending details |
| 4 | Engineering design | 10% | [Open project](projects/04_engineering_design/) | Pending details |
| 5 | Optimal control and reinforcement learning | 10% | [Open project](projects/05_optimal_control_rl/) | Pending details |

Each project folder is prepared for notebooks, reusable source code, figures, numerical results, and a final PDF report. Update the status and add a direct report link after each submission is ready.

## Repository structure

~~~text
.
├── assets/                  # OptiForge visual assets
├── common/                  # Reusable optimization and plotting utilities
├── docs/                    # Submission and participation records
├── presentations/           # Slides for the assigned presentation
├── projects/                # Five graded course projects
└── templates/               # Reusable project and LaTeX report templates
~~~

## Quick start

~~~bash
git clone https://github.com/Liukz233/MAE598-Design-Optimization.git
cd MAE598-Design-Optimization

python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Open notebooks with:

~~~bash
jupyter lab
~~~

## Working conventions

1. Keep exploratory work in each project's `notebooks/` directory.
2. Move reusable functions into `src/` or the shared `common/` package.
3. Save publication-ready figures in `figures/` and concise numerical outputs in `results/`.
4. Export the final report as `report/report.pdf`; keeping this path stable keeps its GitHub URL stable.
5. Record final results and reproduction instructions in the project README.
6. Use meaningful commits such as `project-02: add line-search comparison`.

## Reports and presentations

- Start reports from [the LaTeX template](templates/report_template.tex).
- Use [the submission checklist](docs/submission_checklist.md) before sharing a URL.
- Store presentation materials in [presentations](presentations/).
- Keep a lightweight record of participation in [the participation log](docs/participation_log.md).

## Academic integrity and course materials

This repository is for original coursework produced by Team OptiForge. It does not contain exam work, instructor-owned lecture materials, solution sets, or redistributed course content. Generative AI must not be used for course exams unless explicitly permitted by the instructor.
