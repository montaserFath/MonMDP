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


ARCH_LAYER_NAMES = {
    "cnn": CNN_LAYER_NAMES,
    "mlp": MLP_LAYER_NAMES,
    "rl_q": ["Conv 1 (32 ch)", "Conv 2 (64 ch)", "Linear (512 units)"],
    "rl_reward": ["Hidden 1 (128)", "Hidden 2 (64)"],
}


def layer_names(run_dir, arch=None):
    """Layer labels: --arch if given, else the run's args.json `arch` (continual / RL runs), else CNN."""
    if arch is None:
        try:
            with open(os.path.join(run_dir, "args.json")) as f:
                arch = json.load(f).get("arch")
        except FileNotFoundError:
            pass
    return ARCH_LAYER_NAMES.get(arch, CNN_LAYER_NAMES)


def stack_runs(dfs, x_col, col):
    """(x, data): data is (n_runs, len(x)), NaN where a run has no row at that x (runs may differ in length)."""
    x = np.array(sorted(set().union(*[set(df[x_col]) for df in dfs])))
    return x, np.stack([df.set_index(x_col)[col].reindex(x).values for df in dfs])


def mean_ci(data):
    """Per-column mean and 95% t half-width over the runs that have a value (NaN-aware), plus run counts."""
    n_valid = np.sum(~np.isnan(data), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(data, axis=0)
        sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(n_valid)
        return mean, sem * st.t.ppf(0.975, n_valid - 1), n_valid


def seeds_label(n_valid):
    lo, hi = int(n_valid.min()), int(n_valid.max())
    return f"{hi} seeds" if lo == hi else f"{lo}-{hi} seeds"


def plot_by_layer(dfs, out_dir, names, x_col, xlabel, dormant_ymax):
    """One figure per metric with one mean +- 95% CI line per layer."""
    for prefix, (label, ylim) in LAYER_METRICS.items():
        if ylim == (0, 50):
            ylim = (0, dormant_ymax)
        n_layers = sum(f"{prefix}{i}" in dfs[0] for i in range(10))  # layers actually logged
        if n_layers == 0:
            continue  # e.g. reward-error CSVs have no per-layer columns
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(list(names)[:n_layers] + [f"Layer {j + 1}" for j in range(len(names), n_layers)]):
            x, data = stack_runs(dfs, x_col, f"{prefix}{i}")
            mean, ci, n_valid = mean_ci(data)
            ax.plot(x, mean, lw=3, color=f"C{i + 1}", label=name)
            ax.fill_between(x, np.maximum(mean - ci, 0), mean + ci, color=f"C{i + 1}", alpha=0.2)
        ax.set(xlabel=xlabel, ylabel=label, ylim=ylim)
        ax.set_title(f"Mean of {seeds_label(n_valid)}, 95% CI")
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
    parser.add_argument("--arch", choices=sorted(ARCH_LAYER_NAMES), default=None, help="layer names (default: from args.json)")
    parser.add_argument("--x_max", type=float, default=None, help="drop rows with x_col above this value")
    parser.add_argument("--cols", nargs="+", default=None, help="plot these columns instead of the default metric list")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfs = [pd.read_csv(os.path.join(r, args.csv)) for r in args.runs]
    if args.x_max is not None:
        dfs = [df[df[args.x_col] <= args.x_max + 1e-9] for df in dfs]
    plot_by_layer(dfs, args.out_dir, layer_names(args.runs[0], args.arch), args.x_col, args.xlabel, args.pct_ymax)
    metrics = METRICS if args.cols is None else {c: (c, None) for c in args.cols}
    for col, (label, ylim) in metrics.items():
        if col not in dfs[0]:
            continue  # e.g. online_acc only exists in continual runs
        if col in DORMANT_COLS:
            ylim = (0, args.pct_ymax)
        x, data = stack_runs(dfs, args.x_col, col)
        mean, ci, n_valid = mean_ci(data)
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in data:
            ax.plot(x, s, lw=0.6, alpha=0.5, color="C0")
        ax.plot(x, mean, lw=3, color="C0", label=f"Mean ({seeds_label(n_valid)})")
        ax.fill_between(x, np.maximum(mean - ci, 0), mean + ci, color="C0", alpha=0.25, label="95% CI (t)")
        ax.set(xlabel=args.xlabel, ylabel=label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{col}.png"), dpi=150)
        plt.close(fig)
        print(f"{col}: start {mean[0]:.3g} -> end {mean[-1]:.3g} (end uses {n_valid[-1]} seeds)")


if __name__ == "__main__":
    main()
