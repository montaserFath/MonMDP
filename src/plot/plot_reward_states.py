"""Plot the reward model's predicted reward for the water action in each state against the oracle value.

Usage: python src/plot_reward_states.py results/reward_error/seed_0 ... --out results/reward_error/plots/water_by_state.png
Reads <seed_dir>/reward_error_detail.csv (see src/reward_error.py). Mean over seeds with a 95% t-interval.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="results/reward_error/plots/water_by_state.png")
    parser.add_argument("--x_max", type=float, default=15.0, help="max timesteps (x10^6); 15 = checkpoint 30")
    parser.add_argument("--action", default="water")
    args = parser.parse_args()

    det = pd.concat([pd.read_csv(os.path.join(r, "reward_error_detail.csv")) for r in args.runs])
    det = det[(det.action == args.action) & (det.timesteps_M <= args.x_max + 1e-9)]
    det["label"] = det.kind + det.dryness.map(lambda d: f" (dryness {d:g})")
    n_seeds = det.seed.nunique()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, g) in enumerate(det.groupby("label")):
        piv = g.pivot(index="seed", columns="timesteps_M", values="pred")
        x = piv.columns.values
        mean = piv.mean(axis=0).values
        n = piv.notna().sum(axis=0).values
        ci = piv.sem(axis=0).values * st.t.ppf(0.975, n - 1)
        ax.plot(x, mean, lw=2.5, color=f"C{i}", label=f"{label}: model")
        ax.fill_between(x, mean - ci, mean + ci, color=f"C{i}", alpha=0.2)
        ax.axhline(g["true"].iloc[0], color=f"C{i}", ls="--", lw=1)  # env reward, deliberately not in the legend
    ax.set(xlabel="Training timesteps (x10^6)", ylabel=f"Predicted reward for '{args.action}'")
    ax.set_title(f"Reward model (solid) vs env reward (dashed), mean of {n_seeds} seeds, 95% CI", fontsize=10)
    ax.grid(axis="y")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()
