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

| Project | Topic | Weight | Workspace | Public Markdown report | Status |
|---:|---|---:|---|---|---|
| 1 | Problem formulation | 5% | [Workspace](projects/01_problem_formulation/) | [Open report](projects/01_problem_formulation/report/report.md) | In progress |
| 2 | Gradient descent | 5% | [Workspace](projects/02_gradient_descent/) | [Open template](projects/02_gradient_descent/report/report.md) | Awaiting brief |
| 3 | Neural operator | 10% | [Workspace](projects/03_neural_operator/) | [Open template](projects/03_neural_operator/report/report.md) | Awaiting brief |
| 4 | Engineering design | 10% | [Workspace](projects/04_engineering_design/) | [Open template](projects/04_engineering_design/report/report.md) | Awaiting brief |
| 5 | Optimal control and reinforcement learning | 10% | [Workspace](projects/05_optimal_control_rl/) | [Open template](projects/05_optimal_control_rl/report/report.md) | Awaiting brief |

Project 1 explicitly requires a single publicly accessible Markdown report in the team repository. This repository therefore uses `report/report.md` as the stable primary report path for every project unless a later official brief specifies a different format. PDF exports are optional supplementary artifacts.

## Repository structure

~~~text
.
├── assets/                  # OptiForge visual assets
├── common/                  # Reusable optimization and plotting utilities
├── docs/                    # Submission and participation records
├── presentations/           # Slides for the assigned presentation
├── projects/                # Five graded course projects
└── templates/               # Reusable Markdown and optional LaTeX templates
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

1. Read the official assignment page before editing a project template.
2. Transcribe its required sections and rubric into the project README.
3. Keep the primary graded narrative in `report/report.md`.
4. Keep exploratory work in `notebooks/`; move reusable functions into `src/` or `common/`.
5. Save report-ready figures in `figures/` and concise numerical outputs in `results/`.
6. Test Markdown math, relative image paths, and report links directly on GitHub.
7. Use meaningful commits such as `project-02: add line-search comparison`.

## Reports and presentations

- Start a future assignment from the [Markdown report template](templates/report_template.md).
- Use the [project README template](templates/project_readme_template.md) to map official requirements to files.
- The [LaTeX template](templates/report_template.tex) is retained only for optional PDF export.
- Use the [submission checklist](docs/submission_checklist.md) before sharing a URL.
- Store presentation materials in [presentations](presentations/).

## Academic integrity and course materials

This repository is for original coursework produced by Team OptiForge. It does not contain exam work, instructor-owned lecture materials, solution sets, or redistributed course content. Generative AI must not be used for course exams unless explicitly permitted by the instructor.
