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
def full(half):
    """Mirror a field given on the upper half (rows from the centre line upwards)."""
    return np.vstack([half[::-1], half])


def fig_designs(R):
    c = R["config"]
    L, H = c["length_mm"], c["half_height_mm"]
    cases = [("linear", "Optimized with linear analysis"), ("optimized", "Optimized with large-displacement analysis")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), constrained_layout=True)
    for ax, (key, title) in zip(axes, cases):
        d = R["designs"][key]
        m = d["metrics"]
        rho = np.array(d["rho"]).reshape(c["ny"], c["nx"])
        im = ax.imshow(full(rho), origin="lower", extent=[0, L, -H, H], cmap="Greys", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{title}\nclosing {m['closing_mm']:.2f} mm, stroke {m['stroke_mm']:.2f} mm, "
                     f"stress {m['stress_MPa']:.1f} MPa, material {100 * m['volume_fraction']:.0f}%", fontsize=9.5)
        ax.set_xlabel("x (mm)")
        ax.grid(False)
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=axes, label="Density", shrink=0.7)
    fig.savefig(SITE / "assets" / "fig_designs.png", dpi=170)
    plt.close(fig)


def fig_nonlinear(R):
    c, p = R["config"], R["load_path"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True, sharey=True)
    for k, (key, title) in enumerate((("linear_optimum", "Layout from linear analysis"),
                                      ("optimized", "Reported layout"))):
        ax[k].plot(p[key]["nonlinear"]["closing_mm"], p["force_N"], "o-", ms=3, color=C_MAIN,
                   label="closing, large displacement")
        ax[k].plot(p[key]["linear"]["closing_mm"], p["force_N"], "--", color=C_MAIN, label="closing, linear")
        ax[k].plot(p[key]["nonlinear"]["stroke_mm"], p["force_N"], "s-", ms=3, color=C_SECOND,
                   label="stroke, large displacement")
        ax[k].plot(p[key]["linear"]["stroke_mm"], p["force_N"], "--", color=C_SECOND, label="stroke, linear")
        ax[k].axhline(c["F_in_N"], color=C_REF, lw=1, ls=":")
        ax[k].axvline(c["stroke_max_mm"], color=C_REF, lw=1, ls=":")
        ax[k].set(xlabel="Displacement (mm)", title=title)
    ax[0].set_ylabel("Actuator force (N)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(SITE / "assets" / "fig_nonlinear.png", dpi=170)
    plt.close(fig)


def fig_verification(R):
    v = R["verification"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    h, e = np.array(v["convergence"]["h"]), np.array(v["convergence"]["error"])
    ax[0].loglog(h, e, "o-", color=C_MAIN, label=f"solver, slope {v['convergence']['slope']:.2f}")
    ax[0].loglog(h, e[-1] * (h / h[-1]) ** 2, "--", color=C_REF, label="slope 2")
    ax[0].set(xlabel="Element size / beam depth", ylabel="Relative error of the deflection", title="V2  Pure bending")
    ax[0].set_xticks(h, [f"1/{round(1 / x)}" for x in h], fontsize=8)
    ax[0].minorticks_off()
    ax[0].legend(frameon=False)
    r = np.array(v["newton"]["residuals"])
    ax[1].semilogy(range(len(r)), r, "o-", color=C_MAIN)
    ax[1].set(xlabel="Newton iteration", ylabel="Relative residual", title="V5  Newton, from 90% to full force")
    s = np.array(v["taylor"]["steps"])
    for (key, errs), color in zip(v["taylor"]["errors"].items(), (C_MAIN, C_SECOND, "#7a5aa6")):
        ax[2].loglog(s, errs, "o-", color=color, label=key)
    ax[2].loglog(s[:4], 2 * v["taylor"]["errors"]["objective"][0] * (s[:4] / s[0]) ** 2, "--", color=C_REF,
                 label="slope 2")
    ax[2].set(xlabel=r"Step $\epsilon$", ylabel="Relative error of central difference",
              title="V6  Taylor test, large displacement")
    ax[2].legend(frameon=False, fontsize=8)
    fig.savefig(SITE / "assets" / "fig_verification.png", dpi=170)
    plt.close(fig)


def fig_history(R):
    hist, c = R["history"], R["config"]
    it = np.array(hist["iteration"])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=True)
    ax[0].plot(it, hist["closing_mm"], color=C_MAIN)
    ax[0].set(xlabel="MMA iteration", ylabel="Jaw closing (mm)", title="Objective")
    ax[1].plot(it, np.array(hist["stroke_mm"]) / c["stroke_max_mm"], color=C_MAIN, label="stroke / limit")
    ax[1].plot(it, np.array(hist["stress_MPa"]) / c["sigma_allow_MPa"], color=C_SECOND, label="stress / allowable")
    ax[1].plot(it, np.array(hist["volume_fraction"]) / c["volume_fraction"], color="#7a5aa6", label="material / limit")
    ax[1].axhline(1.0, color=C_REF, ls="--", lw=1)
    ax[1].set(xlabel="MMA iteration", ylabel="Ratio to the limit", title="Constraints (feasible at or below 1)",
              ylim=(0, 2))
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.axvline(hist["stage_change"], color=C_REF, lw=1, ls=":")
    ax[0].text(hist["stage_change"] + 4, min(hist["closing_mm"]) + 0.2, "large-displacement\nanalysis from here",
               fontsize=8, color=C_REF)
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

    def rows_for(designs):
        return [
            ["Jaw closing, objective (mm)"] + [f"{m['closing_mm']:.2f}" for _, m in designs],
            [f"Actuator stroke (mm), limit {c['stroke_max_mm']:.1f}"]
            + [f"{m['stroke_mm']:.2f} {mark(m['pass_stroke'])}" for _, m in designs],
            [f"Peak stress (MPa), limit {c['sigma_allow_MPa']:.1f}"]
            + [f"{m['stress_MPa']:.1f} {mark(m['pass_stress'])}" for _, m in designs],
            [f"Material used (%), limit {100 * c['volume_fraction']:.0f}"]
            + [f"{100 * m['volume_fraction']:.1f} {mark(m['pass_volume'])}" for _, m in designs],
            ["Constraints $g$: stroke / stress / material"]
            + [f"{m['g_stroke']:+.3f} / {m['g_stress']:+.3f} / {m['g_volume']:+.3f}" for _, m in designs],
            ["Grip force on the object (N per jaw)"] + [f"{m['grip_force_N'] + 0.0:.1f}".replace("-0.0", "0.0") for _, m in designs],
            ["Mass (g)"] + [f"{m['mass_g']:.1f}" for _, m in designs],
        ]

    designs = [("Initial, uniform density 0.3", R["designs"]["initial"]["metrics"]),
               ("Optimized", R["designs"]["optimized"]["metrics"])]
    lin = R["models"]["linear"]
    models = [("Linear optimum, as linear analysis sees it", lin["own_model"]),
              ("Linear optimum, large-displacement analysis", lin["nonlinear_model"]),
              ("Large-displacement optimum", R["models"]["nonlinear"]["own_model"]),
              ("Large-displacement optimum, as linear analysis would see it", R["models"]["nonlinear"]["linear_model"])]
    starts = [[s["start"], s["iterations"], f"{s['closing_mm']:.2f}", f"{s['stroke_mm']:.2f}",
               f"{s['stress_MPa']:.1f}", f"{100 * s['volume_fraction']:.0f}", mark(s["feasible"])]
              for s in R["optimization"]["starts"]]
    mesh = [[f"{m['element_mm']:.2f}", f"{m['unknowns']:,}", f"{m['closing_mm']:.2f}", f"{m['stroke_mm']:.2f}",
             f"{m['stress_MPa']:.1f}"] for m in R["optimization"]["mesh_check"]]
    return {
        "ladder": md_table(["", "Test", "Quantity", "Solver", "Exact", "Rel. error", ""], ladder,
                           ["---", "---", "---", "---:", "---:", "---:", "---"]),
        "gradients": md_table(["Check", "Quantity", "Adjoint", "Central difference", "Rel. error", ""], grad,
                              ["---", "---", "---:", "---:", "---:", "---"]),
        "results": md_table([""] + [n for n, _ in designs], rows_for(designs), ["---", "---:", "---:"]),
        "models": md_table([""] + [n for n, _ in models], rows_for(models)[:4], ["---"] + ["---:"] * 4),
        "starts": md_table(["Start and analysis", "Iterations", "Closing (mm)", "Stroke (mm)", "Stress (MPa)",
                            "Material (%)", "All limits met"], starts,
                           ["---", "---:", "---:", "---:", "---:", "---:", "---"]),
        "mesh": md_table(["Element size (mm)", "Unknowns", "Closing (mm)", "Stroke (mm)", "Peak stress (MPa)"], mesh,
                         ["---:"] * 5),
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


VIEWER_MD = ("![Optimized layouts](assets/fig_designs.png)\n\n"
             "*The interactive 3D comparison is on the "
             "[report site](https://liukz233.github.io/MAE598-Design-Optimization/). "
             "This static figure is the one from Section 6.*")
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
