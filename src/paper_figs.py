"""Figures, written to WORK/figures/*.pdf (paths.FIG_DIR) from the results directory and the hardware data.

usage: paper_figs.py [target ...]          (default: every target)

main          main_tfim.pdf: local Bures and trace distance vs Trotter steps, one column per n (uniform TFIM,
              1e5 and 1e4 shots)
geometry      geometry.pdf: state-head ablations and the GNN trunk (gnnb_*.csv of run_gnnb.py, if run) vs Trotter
              steps, block Bures distance in bins of ideal block purity (purity_resolved_B100000.csv of
              paper_review_checks.py purity) and the beta sweep (TFIM n=8, 1e5 shots)
spectral      spectral.pdf: purity, Renyi-2 and concurrence errors vs Trotter steps (TFIM n=8, 1e5 shots)
robust        robust.pdf: size transfer to n=10, training-set size, calibration drift (TFIM, 1e5 shots)
n20           n20_B100000.pdf: 20-qubit chain (tensor-network data), Trotter steps 1-3
hardware      hardware.pdf: ibm_fez runs (results.csv of compare_hw.py) and the quench circuit (blocks CSVs of
              analyze_quench_blocks.py)
unstructured  unstructured_B100000.pdf: local Bures distance for disordered TFIM and random circuits at 1e5 shots
Error bars are standard errors over the test circuits unless stated otherwise.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

R = paths.RES_DIR + os.sep
FIG = paths.FIG_DIR + os.sep
DEPTHS = {"tfim": (1, 2, 3, 4, 6, 8), "dtfim": (1, 2, 3, 4, 6, 8), "random": (4, 8, 12, 16, 20, 24)}
XLAB = {"tfim": "Trotter steps", "dtfim": "Trotter steps", "random": "Circuit depth"}
MLAB = {"local_bures": "Local Bures distance", "local_trace": "Local trace distance",
        "err_purity": "Purity error", "err_renyi2": "Rényi-2 entropy error", "err_concurrence": "Concurrence error"}

ORANGE = (0.84, 0.37, 0.00)
STYLE = {  # key: (label, color, marker, linestyle); baselines cool or grey, ours vermillion
    "Noisy": ("Unmitigated", (0.60, 0.60, 0.60), "o", ":"),
    "ZNE-lin": ("ZNE", (0.00, 0.45, 0.70), "s", "--"),
    "PEC": ("PEC", (0.30, 0.30, 0.30), "*", "--"),
    "CDR": ("CDR", (0.00, 0.62, 0.45), "^", "-."),
    "OLS": ("OLS", (0.45, 0.75, 0.45), "v", ":"),
    "RF": ("RF", (0.80, 0.47, 0.65), "<", ":"),
    "MLP-QEM": ("MLP", (0.55, 0.35, 0.70), ">", ":"),
    "GNN-QEM": ("GNN", (0.15, 0.20, 0.55), "h", "-."),
    "GNN-QEM (Bures)": ("GNN trunk, Bures loss", (0.15, 0.20, 0.55), "h", ":"),
    "BlockGeo-QEM": ("BlockGeo-QEM", ORANGE, "o", "-"),
    "abl: HS state loss": ("HS state loss", (0.95, 0.62, 0.40), "s", "--"),
    "abl: coordinate head only": ("Coordinate head only", (0.55, 0.22, 0.05), "D", ":"),
    "abl: no circuit context": ("No circuit context", (0.98, 0.80, 0.55), "^", "-."),
}

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
                     "ytick.minor.width": 0.4, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
EB = dict(ms=3, capsize=1.8, elinewidth=0.6, capthick=0.6)


def logticks(ax, axis="y"):
    """Label a log axis at 1, 2, 5 x 10^k in plain decimals (only powers of ten if that gives too many)."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    cand = [m * 10.0 ** k for k in range(-6, 2) for m in (1, 2, 5)]
    t = [v for v in cand if lo <= v <= hi]
    if len(t) > 6:
        t = [v for v in t if str(v).lstrip("0.").startswith("1")]
    if len(t) < 3:                                      # narrow range: denser ticks
        cand = [round(m * 10.0 ** k, 12) for k in range(-6, 2) for m in (1, 1.5, 2, 3, 4, 5, 6, 8)]
        t = [v for v in cand if lo <= v <= hi]
        if len(t) > 5:
            t = t[::2]
    a.set_major_locator(matplotlib.ticker.FixedLocator(t))
    a.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in t]))
    a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def read(pattern):
    f = R + pattern
    return pd.read_csv(f) if os.path.exists(f) else None


def series(ax, df, key, met, x="depth", style=None):
    """Mean and standard error of `met` over test circuits for method `key`, plotted against `x`."""
    g = df[df.method == key].groupby(x)[met]
    if g.ngroups == 0:
        return
    mu, se = g.mean(), g.std() / np.sqrt(g.count())
    lab, c, mk, ls = style or STYLE[key]
    ax.errorbar(mu.index, mu.values, yerr=se.values, color=c, marker=mk, ls=ls, label=lab, **EB)


def finish(fig, axes, ncol, name, top=0.5, handles=None):
    h, l = handles or axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=1 - top / fig.get_size_inches()[1])
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + name)
    plt.close(fig)
    print("wrote", FIG + name)


def fig_rows(rows, name, row_h=1.3):
    """Grid with one row per (family, budget, metric, y label) and one column per n (6, 8, 10); legend on top."""
    methods = ["Noisy", "ZNE-lin", "PEC", "CDR", "OLS", "MLP-QEM", "GNN-QEM", "BlockGeo-QEM"]
    ns = (6, 8, 10)
    fig, axes = plt.subplots(len(rows), len(ns), figsize=(7.0, row_h * len(rows) + 0.6), squeeze=False)
    for i, (fam, B, met, ylab) in enumerate(rows):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            base, ours = read(f"base_{fam}_n{n}_B{B}_samples.csv"), read(f"ours_{fam}_n{n}_B{B}_samples.csv")
            if base is not None and ours is not None:
                df = pd.concat([base, ours])
                for m in methods:
                    series(ax, df, m, met)
            ax.set_yscale("log")
            logticks(ax)
            ax.set_xticks(DEPTHS[fam])
            ax.grid(alpha=0.25, lw=0.4)
            if i == 0:
                ax.set_title(f"$n={n}$")
            last_of_family = i == len(rows) - 1 or rows[i + 1][0] != fam and XLAB[rows[i + 1][0]] != XLAB[fam]
            if last_of_family:
                ax.set_xlabel(XLAB[fam])
            if j == 0:
                ax.set_ylabel(ylab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.0)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.subplots_adjust(top=1 - 0.42 / fig.get_size_inches()[1])
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + name)
    plt.close(fig)
    print("wrote", FIG + name)


def fig_n20(B=100000):
    """20-qubit chain (tensor-network data), uniform TFIM at steps 1-3: Bures and trace distance."""
    df = read(f"n20_tfim_B{B}_samples.csv")
    curves = [("Noisy", STYLE["Noisy"]), ("PEC", STYLE["PEC"]), ("CDR", STYLE["CDR"]),
              ("GNN-QEM (transfer 10->20)", ("GNN, $10\\to20$", STYLE["GNN-QEM"][1], "h", ":")),
              ("GNN-QEM", ("GNN", STYLE["GNN-QEM"][1], "h", "-.")),
              ("BlockGeo-QEM (transfer 10->20)", ("BlockGeo-QEM, $10\\to20$", ORANGE, "s", "--")),
              ("BlockGeo-QEM", STYLE["BlockGeo-QEM"])]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), squeeze=False)
    for ax, met, lab in zip(axes[0], ("local_bures", "local_trace"), ("Bures distance", "Trace distance")):
        for key, sty in curves:
            series(ax, df, key, met, style=sty)
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    finish(fig, axes, 3, f"n20_B{B}.pdf", top=0.55)


HW = paths.HW + os.sep


def fig_hardware():
    """ibm_fez: (a) local Bures and (b) trace distance of every method at one and two Trotter steps (means and
    standard errors over 30 test circuits); (c) site-resolved block purity error of the quench circuit versus steps,
    simulation lines (RMS over blocks, mean and sd over 30 shot draws) with hardware points."""
    runs = {1: "ibm_fez_d1_tr100_te30_s1000", 2: "ibm_fez_d2_tr100_te30_s500"}
    df = pd.concat([pd.read_csv(HW + f"{t}/results.csv") for t in runs.values()])
    order = [("Noisy", "Unmitigated"), ("OLS", "OLS"), ("RF", "RF"), ("MLP-QEM", "MLP"), ("GNN-QEM", "GNN"),
             ("abl: HS state loss", "HS control"), ("BlockGeo-QEM", "BlockGeo-QEM")]
    cols = {"Noisy": (0.6, 0.6, 0.6), "OLS": STYLE["OLS"][1], "RF": STYLE["RF"][1], "MLP-QEM": STYLE["MLP-QEM"][1],
            "GNN-QEM": STYLE["GNN-QEM"][1], "abl: HS state loss": (0.95, 0.62, 0.40), "BlockGeo-QEM": ORANGE}
    qb_sim = read("quench_blocks_tfim_n8_B10000.csv")
    qb_hw = pd.read_csv(HW + "quench_ibm_fez/blocks.csv") if os.path.exists(HW + "quench_ibm_fez/blocks.csv") else None
    npanel = 3 if qb_sim is not None else 2
    fig, axes = plt.subplots(1, npanel, figsize=(3.4 if npanel == 2 else 7.0, 1.9), squeeze=False)
    w = 0.11
    for ax, met, lab, tag in zip(axes[0], ("local_bures", "local_trace"), ("Bures distance", "Trace distance"),
                                 ("(a)", "(b)")):
        for k, (key, name) in enumerate(order):
            g = df[df.method == key].groupby("depth")[met]
            x = np.array([1, 2]) + (k - 3) * w
            ax.bar(x, g.mean().values, width=w, color=cols[key], yerr=g.std().values / np.sqrt(g.count().values),
                   error_kw=dict(elinewidth=0.6, capsize=1.5, capthick=0.6), label=name, lw=0)
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xticks([1, 2])
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel(lab)
        ax.set_title(tag, loc="left")
        ax.grid(alpha=0.25, lw=0.4, axis="y")
    if npanel == 3:
        ax = axes[0, 2]
        for key, (name, c, mk, ls) in [("Noisy", ("Unmitigated", (0.6, 0.6, 0.6), "o", ":")),
                                       ("GNN-QEM", ("GNN", STYLE["GNN-QEM"][1], "h", "-.")),
                                       ("BlockGeo-QEM", ("BlockGeo-QEM", ORANGE, "o", "-"))]:
            s = qb_sim[(qb_sim.method == key) & (qb_sim.quantity == "purity")].sort_values("depth")
            ax.errorbar(s.depth, s.rms_over_blocks, yerr=s.sd_of_rms, color=c, marker=mk, ls=ls, **EB)
            if qb_hw is not None:
                h = qb_hw[(qb_hw.method == key) & (qb_hw.quantity == "purity")].sort_values("depth")
                ax.errorbar(h.depth + 0.15, h.rms_over_blocks, yerr=h.sd_of_rms, color=c, marker=mk, ls="none",
                            mfc="white", mew=0.9, **EB)
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xticks([1, 2, 3, 4, 6, 8])
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel("Block purity error")
        ax.set_title("(c)", loc="left")
        ax.grid(alpha=0.25, lw=0.4)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4 if npanel == 2 else 7, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.subplots_adjust(top=1 - (0.52 if npanel == 2 else 0.32) / fig.get_size_inches()[1])
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + "hardware.pdf")
    plt.close(fig)
    print("wrote", FIG + "hardware.pdf")


def fig_geometry(n=8, B=100000):
    """(a, b) state-head ablations and the GNN trunk vs steps (Bures, trace) for n=8; (c) block Bures distance of the HS-trained and
    Bures-trained models in bins of ideal block purity (uniform TFIM n=6, 8, 10, all depths; paper_review_checks.py);
    (d) beta sweep (Bures) at steps 1, 3, 6."""
    ours, base = read(f"ours_tfim_n{n}_B{B}_samples.csv"), read(f"base_tfim_n{n}_B{B}_samples.csv")
    gnnb = read(f"gnnb_tfim_n{n}_B{B}_samples.csv")                 # GNN trunk with the Bures heads, if run
    df = pd.concat([ours, base] + ([gnnb] if gnnb is not None else []))
    beta = read(f"beta_tfim_n{n}_B{B}_samples.csv")
    b1 = ours[ours.method == "BlockGeo-QEM"].assign(method="beta=1")
    beta = pd.concat([beta, b1])
    beta["b"] = beta.method.str.split("=").str[1].astype(float)
    fig = plt.figure(figsize=(3.4, 3.9))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.55, top=0.84, bottom=0.1, left=0.165, right=0.98)
    abl = ["GNN-QEM", "abl: HS state loss", "abl: coordinate head only", "abl: no circuit context", "BlockGeo-QEM"]
    if gnnb is not None:
        abl.insert(1, "GNN-QEM (Bures)")
    short = {"local_bures": "Bures distance", "local_trace": "Trace distance"}
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax, met, tag in zip(axs[:2], ("local_bures", "local_trace"), ("(a)", "(b)")):
        for m in abl:
            series(ax, df, m, met)
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xticks(DEPTHS["tfim"])
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel(short[met])
        ax.set_title(tag, loc="left")
        ax.grid(alpha=0.25, lw=0.4)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0), handlelength=2.4,
               columnspacing=1.2)
    # (c) purity-resolved block Bures distance
    ax = axs[2]
    pr = read(f"purity_resolved_B{B}.csv")
    pr = pr[pr.family == "tfim"].copy()
    bins = [0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0001]
    pr["bin"] = pd.cut(pr.purity, bins, right=False)
    g = pr.groupby("bin", observed=True)
    x = g.purity.mean()
    for col, key in (("hs", "abl: HS state loss"), ("ours", "BlockGeo-QEM")):
        lab, c, mk, ls = STYLE[key]
        ax.errorbar(x.values, g[col].mean().values, yerr=(g[col].std() / np.sqrt(g[col].count())).values,
                    color=c, marker=mk, ls=ls, **EB)
    ax.set_xlabel("Ideal block purity")
    ax.set_ylabel("Block Bures distance")
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(bottom=0)
    ax.set_title("(c)", loc="left")
    ax.grid(alpha=0.25, lw=0.4)
    # (d) beta sweep
    ax = axs[3]
    shades = {1: (0.84, 0.37, 0.00), 3: (0.55, 0.22, 0.05), 6: (0.30, 0.12, 0.02)}
    marks = {1: "o", 3: "s", 6: "D"}
    for d in (1, 3, 6):
        gb = beta[beta.depth == d].groupby("b")["local_bures"]
        mu, se = gb.mean(), gb.std() / np.sqrt(gb.count())
        ax.errorbar(mu.index, mu.values, yerr=se.values, color=shades[d], marker=marks[d], ls="-",
                    label=f"{d}", **EB)
    ax.set_xscale("log")
    ax.set_yscale("log")
    logticks(ax)
    ax.set_xticks([0.1, 0.3, 1, 3, 10])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"Weight $\beta$")
    ax.set_ylabel("Bures distance")
    ax.set_title("(d)", loc="left")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.55, 1.06), ncol=3, frameon=False, handlelength=1.6,
              columnspacing=0.8, fontsize=6, title="Trotter steps", title_fontsize=6)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + "geometry.pdf")
    plt.close(fig)
    print("wrote", FIG + "geometry.pdf")


def fig_spectral(n=8, B=100000):
    base, ours = read(f"basenl_tfim_n{n}_B{B}_samples.csv"), read(f"oursnl_tfim_n{n}_B{B}_samples.csv")
    df = pd.concat([base, ours])
    methods = ["OLS", "MLP-QEM", "GNN-QEM", "abl: HS state loss", "abl: coordinate head only", "BlockGeo-QEM"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.75), squeeze=False)
    for ax, met in zip(axes[0], ("err_purity", "err_renyi2", "err_concurrence")):
        for m in methods:
            series(ax, df, m, met)
        ax.set_yscale("log")
        logticks(ax)
        ax.set_xticks(DEPTHS["tfim"])
        ax.set_xlabel("Trotter steps")
        ax.set_ylabel(MLAB[met])
        ax.grid(alpha=0.25, lw=0.4)
    finish(fig, axes, 6, "spectral.pdf", top=0.2)


def fig_robust(B=100000, d=3):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), squeeze=False)
    ax = axes[0, 0]
    t8, t6 = read(f"transfer_tfim_n8to10_B{B}_samples.csv"), read(f"transfer_tfim_n6to10_B{B}_samples.csv")
    base10, ours10 = read(f"base_tfim_n10_B{B}_samples.csv"), read(f"ours_tfim_n10_B{B}_samples.csv")
    curves = [(base10, "GNN-QEM", ("GNN, trained at $n=10$", STYLE["GNN-QEM"][1], "h", "-.")),
              (t8, "GNN-QEM (transfer)", ("GNN, $8\\to10$", STYLE["GNN-QEM"][1], "h", ":")),
              (t8, "OLS-pooled (transfer)", ("OLS, $8\\to10$", STYLE["OLS"][1], "v", ":")),
              (ours10, "BlockGeo-QEM", ("BlockGeo-QEM, trained at $n=10$", ORANGE, "o", "-")),
              (t8, "BlockGeo-QEM (transfer)", ("BlockGeo-QEM, $8\\to10$", ORANGE, "s", "--")),
              (t6, "BlockGeo-QEM (transfer)", ("BlockGeo-QEM, $6\\to10$", (0.95, 0.62, 0.40), "D", ":"))]
    for df, key, sty in curves:
        if df is not None:
            series(ax, df, key, "local_bures", style=sty)
    ax.set_xticks(DEPTHS["tfim"])
    ax.set_xlabel("Trotter steps")
    ax.set_title("(a) Size transfer to $n=10$", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, handlelength=2.2,
              columnspacing=0.6, fontsize=6)
    ax = axes[0, 1]
    ds = read(f"datasize_tfim_n8_B{B}_samples.csv")
    ds = ds[ds.depth == d]
    for m in ["OLS", "RF", "MLP-QEM", "GNN-QEM", "BlockGeo-QEM"]:
        series(ax, ds, m, "local_bures", x="n_train")
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 400])
    ax.set_xticklabels(["25", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.set_xlabel("Training circuits")
    ax.set_title(f"(b) Training-set size, $n=8$, {d} steps", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False, handlelength=2.2,
              columnspacing=0.6, fontsize=6)
    ax = axes[0, 2]
    dr = read(f"drift_tfim_n8_B{B}_samples.csv")
    dr = dr[dr.depth == d]
    for m in ["Noisy", "ZNE-lin", "PEC", "CDR", "GNN-QEM", "BlockGeo-QEM"]:
        series(ax, dr, m, "local_bures", x="drift")
    ax.set_xticks([0.7, 0.85, 1.0, 1.15, 1.3])
    ax.set_xlabel("Gate-error scale at test time")
    ax.set_title(f"(c) Calibration drift, $n=8$, {d} steps", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False, handlelength=2.2,
              columnspacing=0.6, fontsize=6)
    for ax in axes[0]:
        ax.set_yscale("log")
        logticks(ax)
        ax.set_ylabel(MLAB["local_bures"])
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(FIG + "robust.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", FIG + "robust.pdf")


TARGETS = ("main", "geometry", "spectral", "robust", "n20", "hardware", "unstructured")

if __name__ == "__main__":
    which = sys.argv[1:] or list(TARGETS)
    unknown = [w for w in which if w not in TARGETS]
    if unknown:
        sys.exit(f"unknown target(s) {unknown}; choose from: {' '.join(TARGETS)}")
    if "main" in which:
        fig_rows([("tfim", 100000, "local_bures", "Bures, $10^5$ shots"),
                  ("tfim", 100000, "local_trace", "Trace, $10^5$ shots"),
                  ("tfim", 10000, "local_bures", "Bures, $10^4$ shots")], "main_tfim.pdf")
    if "geometry" in which:
        fig_geometry()
    if "spectral" in which:
        fig_spectral()
    if "robust" in which:
        fig_robust()
    if "n20" in which:
        fig_n20()
    if "hardware" in which:
        fig_hardware()
    if "unstructured" in which:
        fig_rows([("dtfim", 100000, "local_bures", "Bures, disordered"),
                  ("random", 100000, "local_bures", "Bures, random")], "unstructured_B100000.pdf")
