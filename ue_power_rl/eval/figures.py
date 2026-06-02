"""
Generate all paper figures in IEEE double-column format.

Fig 1: Mean UE power (mW) vs traffic load rho — 4 policy curves
Fig 2: EE convergence during training (from training log CSV)
Fig 3: Latency CDF — 4 policies
Fig 4: Ablation — joint optimisation vs single-parameter variants

Output: results/figures/fig{1..4}.pdf  (vector, IEEE-ready)
        results/figures/fig{1..4}.png  (300 dpi, for review)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------- #
# IEEE double-column style
# --------------------------------------------------------------------------- #
IEEE_COL_WIDTH = 3.5      # inches (single column)
IEEE_DPI       = 300

COLORS = {
    "DQN (ours)": "#1a6eb5",
    "3GPP DRX":   "#e07b39",
    "Always-on":  "#4daf4a",
    "Random":     "#984ea3",
    "oracle":     "#a65628",
}
LINESTYLES = {
    "DQN (ours)": "-",
    "3GPP DRX":   "--",
    "Always-on":  "-.",
    "Random":     ":",
}
MARKERS = {
    "DQN (ours)": "o",
    "3GPP DRX":   "s",
    "Always-on":  "^",
    "Random":     "D",
}

def _ieee_fig(nrows=1, ncols=1, width=IEEE_COL_WIDTH, height=None):
    if height is None:
        height = width * 0.75
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         8,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "lines.linewidth":   1.2,
        "lines.markersize":  4,
        "axes.linewidth":    0.6,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.4,
    })
    fig, ax = plt.subplots(nrows, ncols,
                            figsize=(width * ncols, height * nrows))
    return fig, ax


def _save(fig, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png"]:
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight")
    print(f"  Saved {name}")


# --------------------------------------------------------------------------- #
# Figure 1: Power vs traffic load
# --------------------------------------------------------------------------- #
def fig1_power_vs_load(all_results: dict, out_dir: str):
    """
    For each policy, bin episodes by mean traffic load (rho) and
    plot mean UE power in mW.
    """
    fig, ax = _ieee_fig()

    rho_bins = np.linspace(0, 1, 11)
    rho_centres = 0.5 * (rho_bins[:-1] + rho_bins[1:])

    for name, df in all_results.items():
        # reconstruct per-slot rho from tput_samples as proxy
        all_powers  = []
        all_rhos    = []
        for _, row in df.iterrows():
            ps = np.array(row["power_samples"])
            ts = np.array(row["tput_samples"])
            rho_proxy = np.clip(ts / 20.0, 0, 1)    # offered/peak
            all_powers.append(ps)
            all_rhos.append(rho_proxy)

        all_powers = np.concatenate(all_powers)
        all_rhos   = np.concatenate(all_rhos)

        bin_means = []
        for lo, hi in zip(rho_bins[:-1], rho_bins[1:]):
            mask = (all_rhos >= lo) & (all_rhos < hi)
            bin_means.append(np.mean(all_powers[mask]) if mask.sum() > 0 else np.nan)

        ax.plot(rho_centres, bin_means,
                color=COLORS[name], ls=LINESTYLES[name],
                marker=MARKERS[name], markevery=2, label=name)

    ax.set_xlabel("Traffic load $\\rho$")
    ax.set_ylabel("Mean UE power (mW)")
    ax.set_xlim(0, 1)
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig1_power_vs_load", out_dir)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: EE convergence
# --------------------------------------------------------------------------- #
def fig2_ee_convergence(log_csv: str, out_dir: str):
    """
    Smooth training EE curve from the training log.
    Shows convergence of the DQN agent.
    """
    if not os.path.exists(log_csv):
        print(f"  Training log not found: {log_csv} — skipping Fig 2")
        return

    df  = pd.read_csv(log_csv)
    fig, ax = _ieee_fig()

    steps = df["step"].values
    ee    = df["ep_ee_gbits_j"].values

    # rolling mean (window = 20 episodes)
    ee_smooth = pd.Series(ee).rolling(20, min_periods=1).mean().values

    ax.plot(steps / 1e3, ee, alpha=0.25,
            color=COLORS["DQN (ours)"], linewidth=0.6)
    ax.plot(steps / 1e3, ee_smooth,
            color=COLORS["DQN (ours)"], label="DQN (ours, smoothed)")

    # mark 3GPP DRX level if available (horizontal dashed line)
    # placeholder — fill in after evaluation
    ax.axhline(y=0.45, color=COLORS["3GPP DRX"], ls="--",
               linewidth=0.9, label="3GPP DRX (eval)")

    ax.set_xlabel("Training steps (×10³)")
    ax.set_ylabel("Episode EE (Gbits/J)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig2_ee_convergence", out_dir)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: Latency CDF
# --------------------------------------------------------------------------- #
def fig3_latency_cdf(all_results: dict, out_dir: str):
    fig, ax = _ieee_fig()

    for name, df in all_results.items():
        # pool latency samples from all episodes
        all_lat = np.concatenate([np.array(row["lat_samples"])
                                   for _, row in df.iterrows()])
        # clip extreme outliers for display
        all_lat = np.clip(all_lat, 0, 50)
        sorted_lat = np.sort(all_lat)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

        ax.plot(sorted_lat, cdf,
                color=COLORS[name], ls=LINESTYLES[name], label=name)

    # QoS target line
    ax.axvline(x=10, color="black", ls=":", linewidth=0.8,
               label="$D_{max}$ = 10 ms")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig3_latency_cdf", out_dir)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: Ablation bar chart
# --------------------------------------------------------------------------- #
def fig4_ablation(ablation_results: dict, out_dir: str):
    """
    ablation_results: dict name -> mean EE (Gbits/J)
    e.g. {'Joint (ours)': 1.12, 'BWP only': 0.71, 'WUR only': 0.58,
           'SSB only': 0.52, '3GPP DRX': 0.45}
    """
    if not ablation_results:
        print("  No ablation data — skipping Fig 4")
        return

    fig, ax = _ieee_fig()

    names  = list(ablation_results.keys())
    values = [ablation_results[n]["ee"] for n in names]
    errors = [ablation_results[n].get("std", 0) for n in names]
    colors = [COLORS.get(n, "#888888") for n in names]

    bars = ax.bar(range(len(names)), values, yerr=errors,
                   color=colors, width=0.55, capsize=3,
                   error_kw={"linewidth": 0.8})

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Energy efficiency (Gbits/J)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, axis="y")

    # annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    _save(fig, "fig4_ablation", out_dir)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary table (for paper Table III)
# --------------------------------------------------------------------------- #
def print_summary_table(all_results: dict):
    print("\n" + "="*80)
    print(f"{'Policy':<18} | {'EE mean':>9} {'±':>2} {'std':>6} | "
          f"{'Pwr(mW)':>8} | {'Lat p95':>8} | {'Viol%':>6} | {'BWP-full%':>9}")
    print("-"*80)
    for name, df in all_results.items():
        print(f"{name:<18} | {df.ee_gbits_j.mean():>9.3f} ± {df.ee_gbits_j.std():>6.3f} | "
              f"{df.mean_power_mw.mean():>8.1f} | "
              f"{df.p95_lat_ms.mean():>8.1f} | "
              f"{df.pct_lat_violated.mean():>6.1f}% | "
              f"{df.bwp_full_frac.mean()*100:>8.1f}%")


if __name__ == "__main__":
    # Example: generate placeholder figures with synthetic data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="results/eval")
    parser.add_argument("--log",      type=str,
                        default="results/logs/train_v10kmh_d200m_s42.csv")
    parser.add_argument("--out",      type=str, default="results/figures")
    args = parser.parse_args()

    # load eval CSVs if present
    policy_names = ["DQN_ours_", "3GPP_DRX_", "Always-on_", "Random_"]
    # figures are generated by running evaluate.py first, then this script
    print("Run evaluate.py first to generate data, then call generate_figures()")
    fig2_ee_convergence(args.log, args.out)
