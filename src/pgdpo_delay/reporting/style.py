"""SINGLE SOURCE for every visual decision in the paper.

Rules (enforced by convention, checked in review):
  1. No figure function may set figsize, fonts, colors, or method order
     directly -- everything comes from here via new_fig()/save_fig().
  2. Training/evaluation code never draws; figures read saved artifacts only.
  3. Method identity (label, color, order, linestyle) is defined ONCE here so
     every panel across P1-P4 renders methods identically.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH_IN = 5.5                     # ICLR text width
FIGSIZE = {
    "panel":  (TEXTWIDTH_IN*0.48, 1.9),    # one of a 2-across row (Figure 2)
    "wide":   (TEXTWIDTH_IN,      2.2),    # full-width single panel
    "square": (TEXTWIDTH_IN*0.48, TEXTWIDTH_IN*0.48),
    "row4":   (TEXTWIDTH_IN,      1.55),   # 4-across strip
}
RC = {
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.2, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
}
# Method identity: label / color / order / linestyle, identical in every panel.
METHODS = {
    "oracle":   dict(label="Oracle",    color="#000000", ls="-",  z=5, order=0),
    "pgdpo":    dict(label="PGDPO",     color="#d62728", ls="-",  z=4, order=1),
    "lstm_dpo": dict(label="LSTM-DPO",  color="#1f77b4", ls="--", z=3, order=2),
    "pdgm":     dict(label="PDGM",      color="#2ca02c", ls="-.", z=2, order=3),
    "nmpc":     dict(label="NMPC",      color="#9467bd", ls=":",  z=2, order=4),
    "ppo":      dict(label="PPO",       color="#7f7f7f", ls=":",  z=1, order=5),
}
PROBLEM_LABEL = {"p1": "P1 (point delay)", "p2": "P2 (network, distributed)",
                 "p3": "P3 (epidemic)", "p4": "P4 (execution)"}
FIG_DIR = Path("outputs/figures")

def new_fig(kind="panel", ncols=1, nrows=1):
    with plt.rc_context(RC):
        w, h = FIGSIZE[kind]
        fig, ax = plt.subplots(nrows, ncols, figsize=(w*ncols if kind == "panel" else w,
                                                      h*nrows))
    for k, v in RC.items():
        matplotlib.rcParams[k] = v
    return fig, ax

def method_kw(m):
    s = METHODS[m]
    return dict(label=s["label"], color=s["color"], linestyle=s["ls"], zorder=s["z"])

def sorted_methods(ms):
    return sorted(ms, key=lambda m: METHODS[m]["order"])

def save_fig(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    import matplotlib.pyplot as _plt; _plt.close(fig)
    return FIG_DIR / f"{name}.pdf"
