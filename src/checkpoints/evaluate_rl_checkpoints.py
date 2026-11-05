"""Plasticity metrics for the saved Q-networks and reward models (checkpoints_{check}/q_network_{seed}, reward_model_{seed}).

Reuses src/plasticity.py. Probe states follow notebooks/plassticity_loss.ipynb:
    obs_env_sample = env_obs[:, sample_index]   # (n_seeds, 6, 11, 11)
    obs_mon_sample = monitor_obs[:, sample_index]  # (n_seeds,)
The Q-network gets the monitor observation concatenated after layer 5 (as in src/network.py); the reward model
reads only the centre pixel of the observation.

Usage:
  python src/evaluate_rl_checkpoints.py --main_dir <dir with checkpoints_*/, env_obs.npy, monitor_obs.npy> \
      --out_dir results/rl_plasticity [--n_seeds 10 --n_checks 23 --sample_index 2 --device mps:0]
Writes <out_dir>/seed_<s>/{q,reward}_metrics.csv (one row per checkpoint) for src/plot_metrics.py, e.g.
  python src/plot_metrics.py <out_dir>/seed_0 ... --csv q_metrics.csv --x_col timesteps_M --xlabel "Training timesteps (x10^6)"
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plasticity import (dormant_units_by_tau, effective_rank, get_relu_features, percent_active_parameters,  # noqa: E402
                            srank, unit_sign_entropy, weight_norm)


def uses_monitor(model: nn.Sequential, obs_env: torch.Tensor) -> bool:
    """True if the first Linear layer expects one extra input (the monitor observation) after the conv features."""
    n_features = model[:5](obs_env).shape[1]
    n_in = model[5].in_features
    if n_in not in (n_features, n_features + 1):
        raise ValueError(f"Q-network layer 5 expects {n_in} inputs but the conv stack gives {n_features}")
    return n_in == n_features + 1


class QWithMonitor(nn.Module):
    """Q-network forward as in src/network.py: monitor observation concatenated after layer 5.

    With obs_mon=None the plain network is used (Q-networks trained without a monitor observation).
    """

    def __init__(self, model: nn.Sequential, obs_mon: torch.Tensor = None):
        super().__init__()
        self.model = model
        self.obs_mon = None if obs_mon is None else obs_mon.reshape(-1, 1)

    def forward(self, obs_env):
        if self.obs_mon is None:
            return self.model(obs_env)
        return self.model[5:](torch.cat((self.model[:5](obs_env), self.obs_mon), 1))


class RewardCenterPixel(nn.Module):
    """Reward model reads the centre pixel (C values) of the observation window."""

    def __init__(self, model: nn.Sequential):
        super().__init__()
        self.model = model

    def forward(self, obs_env):
        c = obs_env.shape[-1] // 2
        return self.model(obs_env[:, :, c, c])


def rl_metrics(wrapped: nn.Module, inner: nn.Module, obs_env: torch.Tensor, taus) -> dict:
    """Dormant %, srank, effective rank, sign entropy, norms. `dormant_pct`/`dormant_pct_layer{i}` are tau=0.025."""
    wrapped.eval()
    by_tau = dormant_units_by_tau(wrapped, obs_env, tuple(sorted(set(taus) | {0.025})))
    dormant, per_layer = by_tau[0.025]
    last_hidden = get_relu_features(wrapped, obs_env)[-1]
    sign_entropy, sign_layers = unit_sign_entropy(wrapped, obs_env)
    metrics = {
        "dormant_pct": dormant,
        "active_params_pct": percent_active_parameters(inner, 1e-3),
        "effective_rank": effective_rank(last_hidden),
        "srank": srank(last_hidden),
        "unit_sign_entropy": sign_entropy,
        "weight_norm": weight_norm(inner),
    }
    metrics.update({f"dormant_pct_tau_{tau:g}": by_tau[tau][0] for tau in taus})
    metrics.update({f"dormant_pct_layer{i}": v for i, v in enumerate(per_layer)})
    metrics.update({f"unit_sign_entropy_layer{i}": v for i, v in enumerate(sign_layers)})
    return metrics


def load_probe_states(states_dir, n, checks, seed=0):
    """Fixed probe set: `n` states drawn at random (fixed seed) from the src/collect_states.py output of the given
    checkpoints, all seeds pooled. Returns windows (n, 6, w, w) and the monitor observation (n,) recorded with them
    (NaN if the files have none). These windows use the training observation encoding, unlike env_obs.npy."""
    windows, monitor = [], []
    for root, _, names in sorted(os.walk(states_dir)):
        for f in sorted(names):
            if f.endswith(".npz") and int(f[len("checkpoint_"):-len(".npz")]) in checks:
                d = np.load(os.path.join(root, f))
                windows.append(d["windows"].reshape(-1, *d["windows"].shape[-3:]).astype(np.float32))
                monitor.append(d["monitor"].reshape(-1).astype(np.float32) if "monitor" in d
                               else np.full(windows[-1].shape[0], np.nan, np.float32))
    if not windows:
        raise FileNotFoundError(f"no checkpoint_{checks}.npz under {states_dir}")
    windows, monitor = np.concatenate(windows), np.concatenate(monitor)
    pick = np.random.default_rng(seed).choice(len(windows), size=min(n, len(windows)), replace=False)
    return windows[pick], monitor[pick]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", required=True)
    parser.add_argument("--out_dir", default="results/rl_plasticity")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_checks", type=int, default=23)
    parser.add_argument("--sample_index", type=int, default=2,
                        help="checkpoint index of the probe observations (notebook uses 2); -1 uses every seed x checkpoint state")
    parser.add_argument("--states_dir", default=None,
                        help="probe on states collected by src/collect_states.py (training encoding) instead of env_obs.npy")
    parser.add_argument("--probe_n", type=int, default=2000)
    parser.add_argument("--probe_checks", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--timesteps_per_check", type=float, default=5e5)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.025, 0.1])
    parser.add_argument("--device", default="mps:0" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.states_dir:
        obs_env, obs_mon = load_probe_states(args.states_dir, args.probe_n, set(args.probe_checks))
    else:
        obs_env = np.squeeze(np.load(os.path.join(args.main_dir, "env_obs.npy")))
        obs_mon = np.squeeze(np.load(os.path.join(args.main_dir, "monitor_obs.npy")))
        if args.sample_index >= 0:
            obs_env, obs_mon = obs_env[:, args.sample_index], obs_mon[:, args.sample_index]
        else:
            obs_env, obs_mon = obs_env.reshape(-1, *obs_env.shape[2:]), obs_mon.reshape(-1)
    obs_env = torch.tensor(obs_env, dtype=torch.float32, device=device)
    obs_mon = torch.tensor(obs_mon, dtype=torch.float32, device=device)
    print(f"probe states: {tuple(obs_env.shape)} (srank / effective rank cannot exceed {len(obs_env)})", flush=True)

    layer_arch = {"q": "rl_q", "reward": "rl_reward"}
    for seed in range(args.n_seeds):
        rows = {"q": [], "reward": []}
        for check in range(args.n_checks):
            ckpt = os.path.join(args.main_dir, f"checkpoints_{check}")
            if not (os.path.exists(os.path.join(ckpt, f"q_network_{seed}"))
                    and os.path.exists(os.path.join(ckpt, f"reward_model_{seed}"))):
                continue  # this seed's run ended before this checkpoint
            q = torch.load(os.path.join(ckpt, f"q_network_{seed}"), map_location=device, weights_only=False)
            r = torch.load(os.path.join(ckpt, f"reward_model_{seed}"), map_location=device, weights_only=False)
            extra = {"seed": seed, "checkpoint": check, "timesteps_M": check * args.timesteps_per_check / 1e6}
            q_mon = obs_mon if uses_monitor(q, obs_env) else None
            if q_mon is not None and torch.isnan(q_mon).any():
                raise ValueError("Q-network takes the monitor observation but the probe states have none (recollect)")
            if seed == 0 and check == 0:
                print(f"Q-network {'takes' if q_mon is not None else 'does not take'} the monitor observation", flush=True)
            rows["q"].append({**extra, **rl_metrics(QWithMonitor(q, q_mon), q, obs_env, args.taus)})
            rows["reward"].append({**extra, **rl_metrics(RewardCenterPixel(r), r, obs_env, args.taus)})
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        for name, data in rows.items():
            pd.DataFrame(data).to_csv(os.path.join(seed_dir, f"{name}_metrics.csv"), index=False)
        with open(os.path.join(seed_dir, "args.json"), "w") as f:  # arch tells src/plot_metrics.py the layer names
            json.dump({**vars(args), "arch": layer_arch["q"], "arch_reward": layer_arch["reward"]}, f, indent=2)
        print(f"seed {seed}: {len(rows['q'])}/{args.n_checks} checkpoints found -> {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
