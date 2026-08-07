"""All paper figures. Every function: (a) takes ONLY saved artifact paths or
the aggregated results.csv, (b) creates axes via style.new_fig, (c) styles
methods via style.method_kw, (d) writes via style.save_fig. No exceptions."""
import csv
import numpy as np
from . import style

def fig_h_refine(csv_path, name="p1_h_refine"):
    """Finite-h alignment floor (P1 diagnostics appendix)."""
    with open(csv_path) as fp:
        rows = list(csv.DictReader(fp))
    h = np.array([float(r["h"]) for r in rows])
    fl = np.array([float(r["floor_rel"]) for r in rows])
    al = np.array([float(r["nrmse_p"]) for r in rows])
    fig, ax = style.new_fig("panel")
    ax.loglog(h, 100*fl, marker="o", ms=3, **style.method_kw("pgdpo"))
    ax.loglog(h, 100*al, marker="s", ms=3, **style.method_kw("lstm_dpo"))
    ax.lines[-2].set_label("Path-A action floor")
    ax.lines[-1].set_label(r"$p$ alignment")
    ax.loglog(h, 100*fl[-1]*(h/h[-1]), color="0.6", lw=0.8, ls="--",
              label=r"$O(h)$ guide")
    ax.set_xlabel(r"step size $h$"); ax.set_ylabel("relative error (%)")
    ax.legend(frameon=False)
    return style.save_fig(fig, name)

def fig_control_paths(t, series, problem, name):
    """Figure-2 style panel: control trajectories on a common noise path.
    series: {method_key: u_array}; identical styling across all problems."""
    fig, ax = style.new_fig("panel")
    for m in style.sorted_methods(series):
        ax.plot(t, series[m], **style.method_kw(m))
    ax.set_xlabel(r"$t/T$"); ax.set_ylabel(r"$u_t$")
    ax.set_title(style.PROBLEM_LABEL.get(problem, problem), loc="left")
    ax.legend(frameon=False, ncols=2)
    return style.save_fig(fig, f"{name}")
