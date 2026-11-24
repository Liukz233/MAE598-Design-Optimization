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
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parents[1]
SITE, REPORT = PROJECT / "site", PROJECT / "report"
C_MAIN, C_SECOND, C_REF = "#b5490f", "#167d8d", "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.18})


# ----- figures ------------------------------------------------------------------
def outline(ax, R):
    for key, style in (("chip", "-"), ("optic", "--")):
        b = R["components"][key]
        ax.add_patch(Rectangle((1e3 * b["x0"], 1e3 * b["y0"]), 1e3 * (b["x1"] - b["x0"]),
                               1e3 * (b["y1"] - b["y0"]), fill=False, ec="k", lw=1.2, ls=style))


def fig_designs(R):
    c = R["config"]
    extent = [0, c["Lx_mm"], 0, c["Ly_mm"]]
    cases = [("optimized", "Nonlinear radiation"), ("linearized", "Radiation linearized at ambient")]
    T_all = np.concatenate([R["designs"][k]["T_C"] for k, _ in cases])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    for col, (key, title) in enumerate(cases):
        d = R["designs"][key]
        t = np.array(d["t_mm"]).reshape(c["tiles_y"], c["tiles_x"])
        T = np.array(d["T_C"]).reshape(c["ny"] + 1, c["nx"] + 1)
        im0 = axes[0, col].imshow(t, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=c["t_max_mm"])
        axes[0, col].set_title(f"{title}: {d['metrics']['mass_g']:.0f} g")
        im1 = axes[1, col].imshow(T, origin="lower", extent=extent, cmap="Oranges",
                                  vmin=T_all.min(), vmax=T_all.max(), interpolation="bilinear")
        axes[1, col].set_title(f"Nonlinear model: module {d['metrics']['T_chip_C']:.1f} °C, "
                               f"transceiver {d['metrics']['T_optic_C']:.1f} °C", fontsize=9.5)
        for ax in axes[:, col]:
            outline(ax, R)
            ax.grid(False)
            ax.set_xlabel("x (mm)")
        axes[0, 0].set_ylabel("y (mm)")
        axes[1, 0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0, :], label="Thickness (mm)", shrink=0.9)
    fig.colorbar(im1, ax=axes[1, :], label="Temperature (°C)", shrink=0.9)
    fig.savefig(SITE / "assets" / "fig_designs.png", dpi=170)
    plt.close(fig)


def fig_verification(R):
    v = R["verification"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    h, e = np.array(v["mms"]["h_mm"]), np.array(v["mms"]["l2_error_K"])
    ax[0].loglog(h, e, "o-", color=C_MAIN, label=f"solver, slope {v['mms']['slope']:.2f}")
    ax[0].loglog(h, e[-1] * (h / h[-1]) ** 2, "--", color=C_REF, label="slope 2")
    ax[0].set(xlabel="Element size (mm)", ylabel="L2 error (K)", title="V2  Manufactured solution")
    ax[0].set_xticks(h, [f"{v:g}" for v in h], fontsize=8)
    ax[0].minorticks_off()
    ax[0].legend(frameon=False)
    r = np.array(v["newton"]["residuals"])
    ax[1].semilogy(range(len(r)), r, "o-", color=C_MAIN)
    ax[1].set(xlabel="Newton iteration", ylabel="Relative residual", title="V5  Newton convergence")
    ax[1].set_xticks(range(len(r)))
    s = np.array(v["taylor"]["steps"])
    for key, color, label in (("full/chip", C_MAIN, "$g$ module"), ("full/optic", C_SECOND, "$g$ transceiver")):
        ax[2].loglog(s, v["taylor"]["errors"][key], "o-", color=color, label=label)
    ax[2].loglog(s[:4], 2 * v["taylor"]["errors"]["full/chip"][0] * (s[:4] / s[0]) ** 2, "--", color=C_REF,
                 label="slope 2")
    ax[2].set(xlabel=r"Step $\epsilon$", ylabel="Relative error of central difference",
              title="V6  Taylor test, nonlinear model")
    ax[2].legend(frameon=False)
    fig.savefig(SITE / "assets" / "fig_verification.png", dpi=170)
    plt.close(fig)


def fig_history(R):
    hist = R["history"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2), constrained_layout=True)
    ax[0].plot(hist["iteration"], hist["mass_g"], color=C_MAIN)
    ax[0].set(xlabel="SLSQP iteration", ylabel="Mass (g)", title="Objective")
    for key, name, color in (("chip", "Power module", C_MAIN), ("optic", "Transceiver", C_SECOND)):
        ax[1].plot(hist["iteration"], hist[f"T_{key}_C"], color=color, label=f"{name} peak")
        ax[1].axhline(R["components"][key]["limit_C"], color=color, ls="--", lw=1)
    ax[1].set(xlabel="SLSQP iteration", ylabel="Temperature (°C)", title="Constraints (dashed: limits)")
    ax[1].legend(frameon=False)
    fig.savefig(SITE / "assets" / "fig_history.png", dpi=170)
    plt.close(fig)


# ----- tables ------------------------------------------------------------------
def md_table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def mark(ok):
    return "✓ pass" if ok else "✗ FAIL"


def tables(R):
    comp = R["components"]
    ladder = [[r["id"], r["test"], r["quantity"] + (f" ({r['unit']})" if r["unit"] else ""),
               f"{r['computed']:.7g}", f"{r['exact']:.7g}", f"{r['rel_error']:.1e}",
               "pass" if r["pass"] else "FAIL"]
              for r in R["verification"]["ladder"]]
    grad = [r[1:] for r in ladder if r[1].startswith("Adjoint vs")]
    designs = [("Initial, uniform 2 mm", R["designs"]["initial"]["metrics"]),
               (f"Uniform {R['uniform_chip_feasible']['thickness_mm']:.2f} mm", R["uniform_chip_feasible"]),
               ("Optimized", R["designs"]["optimized"]["metrics"])]
    results = [["Mass, objective (g)"] + [f"{m['mass_g']:.1f}" for _, m in designs]]
    for key, name in (("chip", "Power module"), ("optic", "Transceiver")):
        results.append([f"{name} peak (°C), limit {comp[key]['limit_C']:.0f}"]
                       + [f"{m[f'T_{key}_C']:.1f} {mark(m[f'g_{key}'] <= 1e-6)}" for _, m in designs])
        results.append([f"Constraint $g$, {name.lower()} (feasible if at most 0)"]
                       + [f"{m[f'g_{key}'] + 0.0:.4f}" if abs(m[f'g_{key}']) > 5e-5 else "0.0000" for _, m in designs])
    results.append(["Mean plate temperature (°C)"] + [f"{m['T_mean_C']:.1f}" for _, m in designs])
    results.append(["Heat leaving by radiation (W)"] + [f"{m['heat_radiation_W']:.1f}" for _, m in designs])
    starts = [[s["start"], s["iterations"], f"{s['mass_g']:.2f}", f"{s['T_chip_C']:.2f}", f"{s['T_optic_C']:.2f}",
               f"{s['max_thickness_difference_mm']:.3f}"] for s in R["optimization"]["starts"]]
    mesh = [[f"{m['element_mm']:g}", m["elements"], f"{m['T_chip_C']:.2f}", f"{m['T_optic_C']:.2f}"]
            for m in R["optimization"]["mesh_check"]]
    models = []
    for key, name in (("full", "Nonlinear, $\\varepsilon\\sigma(T^4-T_\\infty^4)$"),
                      ("linearized", "Linearized, $h_r(T-T_\\infty)$")):
        m = R["models"][key]
        models.append([name, f"{m['mass_g']:.1f}",
                       f"{m['own_model']['T_chip_C']:.1f} / {m['own_model']['T_optic_C']:.1f}",
                       f"{m['nonlinear_model']['T_chip_C']:.1f} / {m['nonlinear_model']['T_optic_C']:.1f}",
                       f"{m['nonlinear_model']['margin_chip_K']:.1f} / {m['nonlinear_model']['margin_optic_K']:.1f}"])
    E = R["earlier"]
    big = lambda v: f"{v:,.0f}" if v < 1e7 else f"{v:.1e}"
    cond = [[name, f"{c['unknowns']:,}", big(c["kappa"]), big(c["kappa_jacobi"]), c["cg_iterations"],
             c["pcg_iterations"]]
            for name, c in ((f"Plate, uniform 2 mm, {E['conditioning']['uniform_coarse']['element_mm']:g} mm elements",
                             E["conditioning"]["uniform_coarse"]),
                            (f"Plate, uniform 2 mm, {E['conditioning']['uniform']['element_mm']:g} mm elements",
                             E["conditioning"]["uniform"]),
                            (f"Plate, optimized, {E['conditioning']['optimized']['element_mm']:g} mm elements",
                             E["conditioning"]["optimized"]))]
    p2 = E["project2"]
    cond.append([r"Project 2 beam, optimized layout, $E_{\min}=10^{-9}$", f"{p2['unknowns']:,}", big(p2["kappa"]),
                 big(p2["kappa_jacobi"]),
                 p2["cg_iterations"] if p2["cg_converged"] else f"not converged in {p2['cg_iterations']:,}",
                 f"{p2['pcg_iterations']:,}"])
    r3 = ["---", "---:", "---:", "---:"]
    return {
        "ladder": md_table(["", "Test", "Quantity", "Solver", "Exact", "Rel. error", ""], ladder,
                           ["---", "---", "---", "---:", "---:", "---:", "---"]),
        "gradients": md_table(["Check", "Quantity", "Adjoint", "Central difference", "Rel. error", ""], grad,
                              ["---", "---", "---:", "---:", "---:", "---"]),
        "conditioning": md_table(["Matrix", "Unknowns", "$\\kappa$", "$\\kappa$ after Jacobi scaling",
                                  "CG iterations", "Jacobi-PCG iterations"], cond,
                                 ["---", "---:", "---:", "---:", "---:", "---:"]),
        "results": md_table([""] + [n for n, _ in designs], results, r3),
        "starts": md_table(["Start", "Iterations", "Mass (g)", "Module peak (°C)", "Transceiver peak (°C)",
                            "Largest thickness difference from the first run (mm)"], starts,
                           ["---", "---:", "---:", "---:", "---:", "---:"]),
        "mesh": md_table(["Element size (mm)", "Elements", "Module peak (°C)", "Transceiver peak (°C)"], mesh,
                         ["---:", "---:", "---:", "---:"]),
        "models": md_table(["Radiation model used by the optimizer", "Optimal mass (g)",
                            "Peaks in that model (°C), module / transceiver",
                            "Peaks of that design in the nonlinear model (°C)",
                            "Margin to the limits (K)"], models, ["---", "---:", "---:", "---:", "---:"]),
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


VIEWER_MD = ("![Optimized thickness and temperature](assets/fig_designs.png)\n\n"
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
