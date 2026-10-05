"""Plot each plasticity metric vs checkpoint (or task) in its own figure: mean over runs (seeds) with a 95% t-interval.

Usage:
  python src/plot_metrics.py results/mnist/<run> [<run> ...] --out_dir results/mnist/plots
  python src/plot_metrics.py results/continual/<run> ... --csv metrics.csv --x_col task --xlabel Task --pct_ymax 100
Default reads <run>/metrics_recomputed.csv (see src/evaluate_checkpoints.py).
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# column -> (title / y label, y-limits or None for automatic)
METRICS = {
    "dormant_pct_tau_0": ("Dead units, tau=0 (%)", (0, 50)),
    "dormant_pct_tau_0.025": ("Dormant units, tau=0.025 (%)", (0, 50)),
    "dormant_pct_tau_0.1": ("Dormant units, tau=0.1 (%)", (0, 50)),
    "srank": ("Stable rank (srank, delta=0.01)", None),
    "effective_rank": ("Effective rank (entropy)", None),
    "unit_sign_entropy": ("Unit sign entropy (bits)", (0, 1)),
    "grad_norm": ("Gradient norm", None),
    "weight_norm": ("Weight norm", None),
    "active_params_pct": ("Active parameters (%)", (0, 100)),
    "online_acc": ("Online accuracy (%)", (0, 100)),  # continual runs only
    "test_acc": ("Test accuracy (%)", (0, 100)),
}
DORMANT_COLS = ("dormant_pct_tau_0", "dormant_pct_tau_0.025", "dormant_pct_tau_0.1")


# metrics logged per layer: column prefix -> (y label, y-limits); columns are <prefix>0, <prefix>1, ...
LAYER_METRICS = {
    "dormant_pct_layer": ("Dormant units, tau=0.025 (%)", (0, 50)),
    "unit_sign_entropy_layer": ("Unit sign entropy (bits)", (0, 1)),
}
CNN_LAYER_NAMES = ["Conv 1 (32 ch)", "Conv 2 (64 ch)", "Linear (128 units)"]
MLP_LAYER_NAMES = ["Hidden 1 (2000)", "Hidden 2 (2000)", "Hidden 3 (2000)"]


def layer_names(run_dir):
    """Layer labels from the run's args.json (continual runs record `arch`); CNN when absent."""
    try:
        with open(os.path.join(run_dir, "args.json")) as f:
            return MLP_LAYER_NAMES if json.load(f).get("arch") == "mlp" else CNN_LAYER_NAMES
    except FileNotFoundError:
        return CNN_LAYER_NAMES


def plot_by_layer(dfs, out_dir, names, x_col, xlabel, dormant_ymax):
    """One figure per metric with one mean +- 95% CI line per layer."""
    n = len(dfs)
    x = dfs[0][x_col].values
    for prefix, (label, ylim) in LAYER_METRICS.items():
        if ylim == (0, 50):
            ylim = (0, dormant_ymax)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(names):
            data = np.stack([df[f"{prefix}{i}"].values for df in dfs])
            mean = data.mean(0)
            ci = st.sem(data, axis=0) * st.t.ppf(0.975, n - 1)
            ax.plot(x, mean, lw=3, color=f"C{i + 1}", label=name)
            ax.fill_between(x, np.maximum(mean - ci, 0), mean + ci, color=f"C{i + 1}", alpha=0.2)
        ax.set(xlabel=xlabel, ylabel=label, ylim=ylim)
        ax.set_title(f"Mean of {n} seeds, 95% CI")
        ax.grid(axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}s_combined.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out_dir", default="results/mnist/plots")
    parser.add_argument("--csv", default="metrics_recomputed.csv")
    parser.add_argument("--x_col", default="epoch")
    parser.add_argument("--xlabel", default="Checkpoint")
    parser.add_argument("--pct_ymax", type=float, default=50, help="upper y-limit of the dormant-unit plots")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfs = [pd.read_csv(os.path.join(r, args.csv)) for r in args.runs]
    n = len(dfs)
    x = dfs[0][args.x_col].values
    plot_by_layer(dfs, args.out_dir, layer_names(args.runs[0]), args.x_col, args.xlabel, args.pct_ymax)
    for col, (label, ylim) in METRICS.items():
        if col not in dfs[0]:
            continue  # e.g. online_acc only exists in continual runs
        if col in DORMANT_COLS:
            ylim = (0, args.pct_ymax)
        data = np.stack([df[col].values for df in dfs])
        mean = data.mean(0)
        ci = st.sem(data, axis=0) * st.t.ppf(0.975, n - 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in data:
            ax.plot(x, s, lw=0.6, alpha=0.5, color="C0")
        ax.plot(x, mean, lw=3, color="C0", label=f"Mean ({n} seeds)")
        ax.fill_between(x, np.maximum(mean - ci, 0), mean + ci, color="C0", alpha=0.25, label="95% CI (t)")
        ax.set(xlabel=args.xlabel, ylabel=label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{col}.png"), dpi=150)
        plt.close(fig)
        print(f"{col}: start {mean[0]:.3g} -> end {mean[-1]:.3g}")


if __name__ == "__main__":
    main()
