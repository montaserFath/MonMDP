"""Scatter of a Q-error metric per checkpoint (mean over seeds, 95% t-interval as error bars).

Usage: python src/plot_q_error.py --runs own=results/q_error pooled=results/q_error_pooled --metric q_mse \
           --out results/q_error/plots/q_mse.png
Each run dir holds seed_*/q_error.csv (see src/q_error.py). Checkpoints are separate evaluations, so the dots are the data; the thin line only guides the eye.
"""
import argparse
import glob
import os

import matplotlib
import pandas as pd
import scipy.stats as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mean_ci(df, metric):
    piv = df.pivot(index="seed", columns="timesteps_M", values=metric)
    n = piv.notna().sum(axis=0).values
    return piv.columns.values, piv.mean(axis=0).values, piv.sem(axis=0).values * st.t.ppf(0.975, n - 1), piv.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="label=dir pairs")
    parser.add_argument("--metric", default="q_mse")
    parser.add_argument("--out", default="results/q_error/plots/q_mse.png")
    parser.add_argument("--x_max", type=float, default=15.0, help="max timesteps (x10^6); 15 = checkpoint 30")
    parser.add_argument("--group_csv", default="q_error_by_state.csv", help="per-group file used with --by_state")
    parser.add_argument("--by_state", action="store_true",
                        help="one series per kind of cell under the agent (q_error_by_state.csv), one panel per run")
    args = parser.parse_args()

    name = args.group_csv if args.by_state else "q_error.csv"
    runs = []
    for item in args.runs:
        label, path = item.split("=")
        df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(path, "seed_*", name)))
        runs.append((label, df[df.timesteps_M <= args.x_max + 1e-9]))

    panels = [[(label, df)] for label, df in runs] if args.by_state else [runs]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), squeeze=False, sharey=True)
    for ax, panel in zip(axes[0], panels):
        series = [(g, d) for _, df in panel for g, d in df.groupby("group")] if args.by_state else panel
        for i, (label, df) in enumerate(series):
            x, mean, ci, n_seeds = mean_ci(df, args.metric)
            ax.errorbar(x, mean, yerr=ci, fmt="none", ecolor=f"C{i}", alpha=0.5, capsize=2)
            ax.plot(x, mean, lw=1, color=f"C{i}", zorder=2)
            ax.scatter(x, mean, s=22, color=f"C{i}", label=label, zorder=3)
        ax.set(xlabel="Training timesteps (x10^6)", ylabel=args.metric)
        ax.set_title(f"{panel[0][0] + ': ' if args.by_state else ''}Q-network vs true Q*, {args.metric}, "
                     f"mean of {n_seeds} seeds, 95% CI", fontsize=9)
        ax.grid(axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
