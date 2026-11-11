"""Reward-model error per checkpoint against the environment's own reward function (the oracle).

The reward model reads the centre pixel [agent, R, G, B, dryness, wall] of the observation window and predicts one
reward per action. The oracle is `PlantsWateringEnv.reward` from src/gym-monitor/gym_monitor/plant_watering_env.py:
each distinct centre pixel found in env_obs.npy is decoded into an env state (plant / cactus / empty cell, dryness),
the env is put in that state, and `env.reward(action)` is called for all 6 actions. The error is therefore exact over
every input the model can receive in this data, not a sample.

Usage:
  python src/reward_error.py --main_dir <dir with checkpoints_*/, env_obs.npy> --out_dir results/reward_error
Writes <out_dir>/oracle_table.csv, <out_dir>/seed_<s>/reward_error.csv (one row per checkpoint) and
reward_error_detail.csv (one row per checkpoint x state x action).
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym-monitor", "gym_monitor", "plant_watering_env.py")
WATER = 4
ACTION_NAMES = ["up", "down", "right", "left", "water", "nothing"]


def load_env_class():
    """Import PlantsWateringEnv from its file (avoids registering the gym_monitor package)."""
    spec = importlib.util.spec_from_file_location("plant_watering_env", ENV_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PlantsWateringEnv


def load_windows(path: str) -> np.ndarray:
    """Observation windows (N, C, H, W) from a .npy file (any leading dims) or from a directory of
    src/collect_states.py output (all <dir>/**/*.npz, key 'windows')."""
    if os.path.isdir(path):
        files = sorted(os.path.join(root, f) for root, _, names in os.walk(path) for f in names if f.endswith(".npz"))
        if not files:
            raise FileNotFoundError(f"no .npz files under {path}")
        arrays = [np.load(f)["windows"] for f in files]
        return np.concatenate([a.reshape(-1, *a.shape[-3:]) for a in arrays]).astype(np.float32)
    obs = np.squeeze(np.load(path))
    return obs.reshape(-1, *obs.shape[-3:])


def decode_state(vec: np.ndarray) -> tuple:
    """Centre pixel [agent, R, G, B, dryness, wall] -> (kind, dryness); kind in {'empty', 'plant', 'cactus'}."""
    agent, r, g, b, dryness, wall = vec
    if agent != 1 or wall != 0:
        raise ValueError(f"unexpected centre pixel (agent must be present, wall absent): {vec}")
    if r == 0 and g == 0 and b == 0:
        return "empty", float(dryness)
    if r > 0 and g > 0 and b == 0:  # (1,1,0), or (0.5,0.5,0) when observations are normalised
        return "plant", float(dryness)
    if r == 0 and g > 0 and b > 0:  # (0,1,1) / (0,.5,.5)
        return "cactus", float(dryness)
    raise ValueError(f"unknown object encoding in centre pixel: {vec}")


def env_reward(env, kind: str, dryness: float, action: int) -> float:
    """Put the env in the state described by (kind, dryness) with the agent at the centre cell, ask env.reward."""
    if kind == "cactus" and not (env.add_new_plants or env.add_more_plants):
        # without cacti the env treats the cell as empty and would silently return -0.2 instead of -1
        raise ValueError("cactus state needs an env built with add_new_plants=True")
    n = env._n_raws
    pos = np.array([n // 2, n // 2])
    far = np.array([[0, 0]])
    env._grid = np.zeros(env._obs_shape)
    env._grid[2, pos[0], pos[1]] = dryness
    env._agent_pos = pos
    env._plants_pos = pos[None] if kind == "plant" else far
    env._new_plants_pos = pos[None] if kind == "cactus" else far
    env._more_plants_pos = far
    return float(env.reward(action))


def group_of(kind: str, dryness: float, action: int, dry_min: float) -> str:
    if action != WATER:
        return "non_water"
    if kind == "plant":
        return "water_dry_plant" if dryness > dry_min else "water_wet_plant"
    return f"water_{kind}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", required=True)
    parser.add_argument("--out_dir", default="results/reward_error")
    parser.add_argument("--obs_file", default=None,
                        help="windows to take the evaluation states from: a .npy file or a src/collect_states.py "
                             "output dir (default: <main_dir>/env_obs.npy)")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_checks", type=int, default=36)
    parser.add_argument("--timesteps_per_check", type=float, default=5e5)
    parser.add_argument("--dry_difference", type=float, default=0.5)
    parser.add_argument("--add_new_plants", action="store_true", help="env has cacti (needed if cactus states occur)")
    parser.add_argument("--device", default="mps:0" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Distinct centre-pixel inputs and how often each occurs in the saved observations
    obs = load_windows(args.obs_file or os.path.join(args.main_dir, "env_obs.npy"))
    c = obs.shape[-1] // 2
    centre = obs[:, :, c, c].round(4)
    states, counts = np.unique(centre, axis=0, return_counts=True)
    weights = counts / counts.sum()

    EnvClass = load_env_class()
    env = EnvClass(grid_size=(5, 5), n_plants=1, plants_dryness_prob=0.0, dry_difference=args.dry_difference,
                   max_episode_steps=1, add_new_plants=args.add_new_plants)
    dry_min = float(np.min(env.dryness_levels))
    kinds = [decode_state(v) for v in states]
    oracle = np.array([[env_reward(env, k, d, a) for a in range(6)] for k, d in kinds])

    os.makedirs(args.out_dir, exist_ok=True)
    table = pd.DataFrame([{"state_id": i, "kind": k, "dryness": d, "frequency": w,
                           **{f"r_{ACTION_NAMES[a]}": oracle[i, a] for a in range(6)}}
                          for i, ((k, d), w) in enumerate(zip(kinds, weights))])
    table.to_csv(os.path.join(args.out_dir, "oracle_table.csv"), index=False)
    print("oracle reward table (from env.reward):\n" + table.round(3).to_string(index=False), flush=True)

    groups = np.array([[group_of(k, d, a, dry_min) for a in range(6)] for k, d in kinds])
    x = torch.tensor(states, dtype=torch.float32, device=device)
    true_water_best = oracle[:, WATER] > np.delete(oracle, WATER, axis=1).max(1)

    for seed in range(args.n_seeds):
        rows, detail = [], []
        for check in range(args.n_checks):
            path = os.path.join(args.main_dir, f"checkpoints_{check}", f"reward_model_{seed}")
            if not os.path.exists(path):
                continue  # this seed's run ended before this checkpoint
            model = torch.load(path, map_location=device, weights_only=False).eval()
            with torch.no_grad():
                pred = model(x).cpu().numpy()
            err = pred - oracle
            pred_water_best = pred[:, WATER] > np.delete(pred, WATER, axis=1).max(1)
            row = {"seed": seed, "checkpoint": check, "timesteps_M": check * args.timesteps_per_check / 1e6,
                   "mse_all": float((err ** 2).mean()),
                   "mse_weighted": float((weights * (err ** 2).mean(1)).sum()),
                   "mae_all": float(np.abs(err).mean()),
                   "max_abs_err": float(np.abs(err).max()),
                   "water_decision_acc": float((pred_water_best == true_water_best).mean()),
                   "water_decision_acc_weighted": float((weights * (pred_water_best == true_water_best)).sum())}
            for g in np.unique(groups):
                row[f"mse_{g}"] = float((err[groups == g] ** 2).mean())
            rows.append(row)
            for i, (k, d) in enumerate(kinds):
                for a in range(6):
                    detail.append({"seed": seed, "checkpoint": check, "timesteps_M": row["timesteps_M"], "state_id": i,
                                   "kind": k, "dryness": d, "action": ACTION_NAMES[a], "true": oracle[i, a],
                                   "pred": float(pred[i, a])})
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "reward_error.csv"), index=False)
        pd.DataFrame(detail).to_csv(os.path.join(seed_dir, "reward_error_detail.csv"), index=False)
        with open(os.path.join(seed_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"seed {seed}: {len(rows)} checkpoints -> {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
