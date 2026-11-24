"""Draw the figures and build the report from results.json.

    python projects/04_engineering_design/src/make_site.py

Reads site/assets/results.json and report/report.template.md. Writes
site/assets/fig_*.png, report/report.md (for reading on GitHub) and
site/index.html (the report site with the 3D viewer). Numbers in the text are
placeholders of the form [[path.to.value|format]] filled from results.json.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SITE, REPORT = PROJECT / "site", PROJECT / "report"
C_MAIN, C_SECOND, C_REF = "#b5490f", "#167d8d", "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.18})


# ----- figures ------------------------------------------------------------------
def full_beam(half):
    """Mirror a half-beam field (rows from the bottom) about midspan."""
    return np.hstack([half[:, ::-1], half])


def fig_designs(R):
    c = R["config"]
    L, H = c["half_span_mm"], c["depth_mm"]
    panels = []
    for key, title in (("optimized", "Project 4, nonlinear material"),
                       ("linear", "Project 4, linear material assumed")):
        d = R["designs"][key]
        m = d["metrics"]
        panels.append((np.array(d["t_mm"]).reshape(c["tiles_y"], c["tiles_x"]),
                       f"{title}: {m['mass_kg']:.2f} kg, deflection {m['deflection_mm']:.2f} mm, "
                       f"stress {m['stress_MPa']:.0f} MPa"))
    p1 = R["project1"]
    panels.append((np.array(p1["solid"]).reshape(40, 120) * p1["thickness_mm"],
                   f"Project 1 layout as a {p1['thickness_mm']:.1f} mm plate of the same mass: deflection "
                   f"{p1['nonlinear']['deflection_mm']:.2f} mm, stress {p1['nonlinear']['stress_MPa']:.0f} MPa"))
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (t, title) in zip(axes, panels):
        im = ax.imshow(full_beam(t), origin="lower", extent=[-L, L, 0, H], cmap="Greys", vmin=0,
                       vmax=c["t_max_mm"], aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("y (mm)")
        ax.grid(False)
    axes[-1].set_xlabel("x (mm), midspan at 0")
    fig.colorbar(im, ax=axes, label="Plate thickness (mm)", shrink=0.6)
    fig.savefig(SITE / "assets" / "fig_designs.png", dpi=170)
    plt.close(fig)


def fig_nonlinear(R):
    c = R["config"]
    E, fy, n = 1e3 * c["E_GPa"], c["f_y_MPa"], c["n"]
    s = np.linspace(0, 1.05 * fy, 200)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    ax[0].plot(100 * (s / E + 0.002 * (s / fy) ** n), s, color=C_MAIN, label="Ramberg–Osgood, $n=6$")
    ax[0].plot(100 * s / E, s, "--", color=C_REF, label="linear, $E$ = 200 GPa")
    ax[0].axhline(c["sigma_allow_MPa"], color=C_SECOND, lw=1, ls=":")
    ax[0].text(0.3, c["sigma_allow_MPa"] + 4, "allowable stress", color=C_SECOND, fontsize=9)
    ax[0].set(xlabel="Strain (%)", ylabel="Stress (MPa)", title="Stainless steel 1.4301 in tension")
    ax[0].legend(frameon=False, loc="lower right")
    p = R["load_path"]
    f = np.array(p["factor"]) * c["P_kN"]
    ax[1].plot(p["nonlinear_mm"], f, "o-", ms=3, color=C_MAIN, label="nonlinear material")
    ax[1].plot(p["linear_mm"], f, "--", color=C_REF, label="linear material")
    ax[1].axvline(c["deflection_limit_mm"], color=C_SECOND, lw=1, ls=":")
    ax[1].axhline(c["P_kN"], color=C_SECOND, lw=1, ls=":")
    ax[1].set(xlabel="Midspan deflection (mm)", ylabel="Load (kN)", title="Optimized beam, load raised to 1.5 P")
    ax[1].legend(frameon=False, loc="lower right")
    fig.savefig(SITE / "assets" / "fig_nonlinear.png", dpi=170)
    plt.close(fig)


def fig_verification(R):
    v = R["verification"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    h, e = np.array(v["convergence"]["h_mm"]), np.array(v["convergence"]["error"])
    ax[0].loglog(h, e, "o-", color=C_MAIN, label=f"solver, slope {v['convergence']['slope']:.2f}")
    ax[0].loglog(h, e[-1] * (h / h[-1]) ** 2, "--", color=C_REF, label="slope 2")
    ax[0].set(xlabel="Element size (mm)", ylabel="Relative error of the deflection", title="V2  Pure bending")
    ax[0].set_xticks(h, [f"{x:g}" for x in h], fontsize=8)
    ax[0].minorticks_off()
    ax[0].legend(frameon=False)
    r = np.array(v["newton"]["residuals"])
    ax[1].semilogy(range(len(r)), r, "o-", color=C_MAIN)
    ax[1].set(xlabel="Newton iteration", ylabel="Relative residual", title="V5  Newton convergence")
    ax[1].set_xticks(range(len(r)))
    s = np.array(v["taylor"]["steps"])
    for key, color, label in (("deflection", C_MAIN, "$g$ deflection"), ("stress", C_SECOND, "$g$ stress")):
        ax[2].loglog(s, v["taylor"]["errors"][key], "o-", color=color, label=label)
    ax[2].loglog(s[:4], 2 * v["taylor"]["errors"]["deflection"][0] * (s[:4] / s[0]) ** 2, "--", color=C_REF,
                 label="slope 2")
    ax[2].set(xlabel=r"Step $\epsilon$", ylabel="Relative error of central difference",
              title="V6  Taylor test, nonlinear material")
    ax[2].legend(frameon=False)
    fig.savefig(SITE / "assets" / "fig_verification.png", dpi=170)
    plt.close(fig)


def fig_history(R):
    hist, c = R["history"], R["config"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=True)
    ax[0].plot(hist["iteration"], hist["mass_kg"], color=C_MAIN)
    ax[0].set(xlabel="SLSQP iteration", ylabel="Mass (kg)", title="Objective")
    ax[1].plot(hist["iteration"], np.array(hist["deflection_mm"]) / c["deflection_limit_mm"], color=C_MAIN,
               label="deflection / limit")
    ax[1].plot(hist["iteration"], np.array(hist["stress_MPa"]) / c["sigma_allow_MPa"], color=C_SECOND,
               label="stress / allowable")
    ax[1].axhline(1.0, color=C_REF, ls="--", lw=1)
    ax[1].set(xlabel="SLSQP iteration", ylabel="Ratio to the limit", title="Constraints (feasible below 1)")
    ax[1].legend(frameon=False)
    fig.savefig(SITE / "assets" / "fig_history.png", dpi=170)
    plt.close(fig)


# ----- tables ------------------------------------------------------------------
def md_table(header, rows, align):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def mark(ok):
    return "✓ pass" if ok else "✗ FAIL"


def tables(R):
    c = R["config"]
    ladder = [[r["id"], r["test"], r["quantity"], f"{r['computed']:.7g}", f"{r['exact']:.7g}",
               f"{r['rel_error']:.1e}", "pass" if r["pass"] else "FAIL"] for r in R["verification"]["ladder"]]
    grad = [r[1:] for r in ladder if r[1].startswith("Adjoint vs")]
    designs = [("Initial, uniform 6 mm", R["designs"]["initial"]["metrics"]),
               (f"Thinnest feasible uniform plate, {R['uniform_feasible']['thickness_mm']:.2f} mm",
                R["uniform_feasible"]),
               ("Optimized", R["designs"]["optimized"]["metrics"])]
    num = lambda v: "0.0000" if abs(v) < 5e-5 else f"{v:.4f}"
    results = [
        ["Mass, objective (kg)"] + [f"{m['mass_kg']:.2f}" for _, m in designs],
        [f"Midspan deflection (mm), limit {c['deflection_limit_mm']:.1f}"]
        + [f"{m['deflection_mm']:.2f} {mark(m['pass_deflection'])}" for _, m in designs],
        ["Constraint $g$, deflection (feasible if at most 0)"] + [num(m["g_deflection"]) for _, m in designs],
        [f"Peak tile stress (MPa), limit {c['sigma_allow_MPa']:.0f}"]
        + [f"{m['stress_MPa']:.0f} {mark(m['pass_stress'])}" for _, m in designs],
        ["Constraint $g$, stress (feasible if at most 0)"] + [num(m["g_stress"]) for _, m in designs],
    ]
    starts = [[s["start"], s["iterations"], f"{s['mass_kg']:.4f}", f"{s['deflection_mm']:.3f}",
               f"{s['stress_MPa']:.1f}", f"{s['max_thickness_difference_mm']:.3f}"]
              for s in R["optimization"]["starts"]]
    mesh = [[f"{m['element_mm']:g}", f"{m['unknowns']:,}", f"{m['deflection_mm']:.3f}", f"{m['stress_MPa']:.2f}"]
            for m in R["optimization"]["mesh_check"]]
    models = []
    for key, name in (("nonlinear", "Ramberg–Osgood"), ("linear", "Linear, $E$ only")):
        m = R["models"][key]
        models.append([name, f"{m['own_model']['mass_kg']:.2f}",
                       f"{m['own_model']['deflection_mm']:.2f} / {m['own_model']['stress_MPa']:.0f}",
                       f"{m['nonlinear_model']['deflection_mm']:.2f} {mark(m['nonlinear_model']['pass_deflection'])}",
                       f"{m['nonlinear_model']['stress_MPa']:.0f} {mark(m['nonlinear_model']['pass_stress'])}"])
    p1, opt = R["project1"], R["designs"]["optimized"]["metrics"]
    limit_d, limit_s = c["deflection_limit_mm"], c["sigma_allow_MPa"]
    compare = [
        ["Project 4 optimum, variable thickness", f"{opt['mass_kg']:.2f}",
         f"{opt['deflection_mm']:.2f} {mark(opt['pass_deflection'])}",
         f"{opt['stress_MPa']:.0f} {mark(opt['pass_stress'])}",
         f"{R['optimization']['linear_analysis_of_optimum']['deflection_mm']:.2f}"],
        [f"Project 1 layout, {p1['thickness_mm']:.1f} mm plate", f"{opt['mass_kg']:.2f}",
         f"{p1['nonlinear']['deflection_mm']:.2f} {mark(p1['nonlinear']['deflection_mm'] <= limit_d)}",
         f"{p1['nonlinear']['stress_MPa']:.0f} {mark(p1['nonlinear']['stress_MPa'] <= limit_s)}",
         f"{p1['linear']['deflection_mm']:.2f}"],
    ]
    r4 = ["---", "---:", "---:", "---:"]
    return {
        "ladder": md_table(["", "Test", "Quantity", "Solver", "Exact", "Rel. error", ""], ladder,
                           ["---", "---", "---", "---:", "---:", "---:", "---"]),
        "gradients": md_table(["Check", "Quantity", "Adjoint", "Central difference", "Rel. error", ""], grad,
                              ["---", "---", "---:", "---:", "---:", "---"]),
        "results": md_table([""] + [n for n, _ in designs], results, r4),
        "starts": md_table(["Start", "Iterations", "Mass (kg)", "Deflection (mm)", "Peak stress (MPa)",
                            "Largest thickness difference from the first run (mm)"], starts,
                           ["---", "---:", "---:", "---:", "---:", "---:"]),
        "mesh": md_table(["Element size (mm)", "Unknowns", "Deflection (mm)", "Peak tile stress (MPa)"], mesh,
                         ["---:", "---:", "---:", "---:"]),
        "models": md_table(["Material model used by the optimizer", "Optimal mass (kg)",
                            "Deflection (mm) / stress (MPa) in that model",
                            "Deflection with the real material (mm)", "Stress with the real material (MPa)"],
                           models, ["---", "---:", "---:", "---:", "---:"]),
        "project1": md_table(["Design", "Mass (kg)", f"Deflection (mm), limit {limit_d:.1f}",
                              f"Peak tile stress (MPa), limit {limit_s:.0f}",
                              "Deflection if the material were linear (mm)"], compare, r4 + ["---:"]),
    }


# ----- filling and rendering -----------------------------------------------------
def lookup(R, path):
    value = R
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def fill(text, R):
    tabs = tables(R)
    text = re.sub(r"\[\[TABLE:(\w+)\]\]", lambda m: tabs[m.group(1)], text)

    def number(m):
        value = lookup(R, m.group(1))
        return format(value, m.group(2)) if m.group(2) else str(value)

    return re.sub(r"\[\[([\w.]+)(?:\|([^\]]+))?\]\]", number, text)


VIEWER_MD = ("![Optimized plate thickness](assets/fig_designs.png)\n\n"
             "*The interactive 3D comparison is on the "
             "[report site](https://liukz233.github.io/MAE598-Design-Optimization/). "
             "This static figure is the one from Section 7.*")
VIEWER_HTML = '<div id="viewer" class="viewer"><p class="v-wait">Loading the 3D comparison…</p></div>'


def to_html(md_text, title):
    """Markdown to HTML with the math kept out of the Markdown parser."""
    stash = []

    def keep(tex, display):
        stash.append(("\\[" + tex.strip() + "\\]") if display else ("\\(" + tex + "\\)"))
        return f"@@MATH{len(stash) - 1}@@"

    text = re.sub(r"```math\n(.*?)\n```", lambda m: "\n" + keep(m.group(1), True) + "\n", md_text, flags=re.S)
    text = re.sub(r"\$([^$\n]+?)\$", lambda m: keep(m.group(1), False), text)
    text = text.replace("<!--VIEWER-->", VIEWER_HTML)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"@@MATH(\d+)@@", lambda m: html.escape(stash[int(m.group(1))], quote=False), body)
    body = body.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
    page = (Path(__file__).parent / "page_template.html").read_text()
    return page.replace("{{title}}", html.escape(title)).replace("{{content}}", body)


def main():
    R = json.loads((SITE / "assets" / "results.json").read_text())
    fig_designs(R)
    fig_nonlinear(R)
    fig_verification(R)
    fig_history(R)
    text = fill((REPORT / "report.template.md").read_text(), R)
    title = re.search(r"^# (.+)$", text, flags=re.M).group(1)
    (SITE / "index.html").write_text(to_html(text, title))
    github = text.replace("<!--VIEWER-->", VIEWER_MD).replace("](assets/", "](../site/assets/")
    (REPORT / "report.md").write_text(github)
    print(f"wrote {SITE / 'index.html'} and {REPORT / 'report.md'}")


if __name__ == "__main__":
    main()
