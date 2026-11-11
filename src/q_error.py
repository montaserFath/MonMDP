"""Q-network error per checkpoint against the true optimal action values Q* (src/true_q.py).

For every state recorded by src/collect_states.py the network sees the 11x11 window and Q* is looked up from the full
state, so the error includes what the window hides (plants outside it) as well as what the network has not learned.
Joint actions are compared as Q*(s, (a, m)) = Q*_mdp(s, a) - monitor_cost * m (index mdp + 6 * m).

Usage:
  python src/q_error.py --main_dir <dir with checkpoints_*/> --states_dir results/states --q_dir results/true_q \
      --out_dir results/q_error [--pool_checks 10 20 30]
By default each checkpoint is evaluated on the states its own policy visited; with --pool_checks every checkpoint is
evaluated on the same pooled states (those checkpoints of all seeds), so that checkpoints are compared on identical inputs.
Writes <out_dir>/seed_<s>/q_error.csv, one row per checkpoint, and q_error_by_state.csv, one row per checkpoint and
kind of cell the agent stands on (dry plant, half-dry plant, watered plant, empty cell).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.checkpoints.evaluate_rl_checkpoints import QWithMonitor, uses_monitor  # noqa: E402
from src.true_q import joint_q, plant_layout, solve_layouts, state_index  # noqa: E402


def true_q_of_states(full, layout_ids, tables, monitor_cost=0.2):
    """Q* (N, 12) of full states (N, 3, n, n) that belong to layouts `layout_ids` (N,) (keys of `tables`).
    monitor_cost=None: no monitor action, so Q* is the (N, 6) MDP value."""
    full = np.asarray(full, dtype=np.float64)
    out = np.zeros((len(full), 6 if monitor_cost is None else 12))
    for layout in np.unique(layout_ids):
        rows = layout_ids == layout
        Q, plants = tables[int(layout)]
        idx, cell = state_index(full[rows], plants)
        q = Q[idx, cell].astype(np.float64)
        out[rows] = q if monitor_cost is None else joint_q(q, monitor_cost)
    return out


def q_metrics(pred, true):
    """Error statistics between predicted and true Q-values, both (N, n_actions) (12 joint actions, or 6 without a monitor action)."""
    err = pred - true
    centred = err - err.mean(1, keepdims=True)  # what is left once each state's mean offset is removed
    v_true = true.max(1)
    greedy = pred.argmax(1)
    chosen = true[np.arange(len(true)), greedy]
    return {
        "q_mse": float((err ** 2).mean()),
        "q_mae": float(np.abs(err).mean()),
        "q_bias": float(err.mean()),  # > 0: over-estimation
        "q_max_abs": float(np.abs(err).max()),
        "q_mse_centred": float((centred ** 2).mean()),  # error in the relative ranking of actions
        "q_nmse": float((err ** 2).mean() / true.var()),
        "v_mae": float(np.abs(pred.max(1) - v_true).mean()),  # error of the greedy value max_a Q(s, a)
        "v_bias": float((pred.max(1) - v_true).mean()),
        "greedy_regret": float((v_true - chosen).mean()),  # value lost by acting greedily on the network
        "greedy_optimal": float((chosen >= v_true - 1e-6).mean()),
        "q_corr": float(np.corrcoef(pred.ravel(), true.ravel())[0, 1]),
        "q_water_mse": float((err[:, 4] ** 2).mean()),  # the action whose reward depends on the cell (index 4 = water, no monitor request)
        "q_water_bias": float(err[:, 4].mean()),
    }


GROUPS = ["dry plant", "half-dry plant", "watered plant", "empty cell"]


def state_groups(full):
    """Kind of cell the agent stands on in full states (N, 3, n, n): plant by dryness (1, 0.5, 0) or an empty cell."""
    full = np.asarray(full, dtype=np.float64)
    flat = full.reshape(len(full), 3, -1)
    cell = flat[:, 0].argmax(1)
    rows = np.arange(len(full))
    plant, dryness = flat[rows, 1, cell] > 0, flat[rows, 2, cell]
    return np.where(~plant, 3, np.where(dryness > 0.75, 0, np.where(dryness > 0.25, 1, 2)))


def state_zones(full, monitor_column_ind=5, wall_cols=2):
    """Zone of the agent in full states (N, 3, n, n): [plant | empty cell] x [left half (monitored) | right | right wall
    (the last `wall_cols` columns)]. Returns indices into the list zone_names(...)."""
    full = np.asarray(full, dtype=np.float64)
    n = full.shape[-1]
    flat = full.reshape(len(full), 3, -1)
    cell = flat[:, 0].argmax(1)
    plant = flat[np.arange(len(full)), 1, cell] > 0
    col = cell % n
    band = np.where(col < monitor_column_ind, 0, np.where(col < n - wall_cols, 1, 2))
    return np.where(plant, 0, 3) + band


def zone_names(monitor_column_ind=5, n=10, wall_cols=2):
    bands = [f"left, monitored (cols 0-{monitor_column_ind - 1})", f"right (cols {monitor_column_ind}-{n - wall_cols - 1})",
             f"right wall (cols {n - wall_cols}-{n - 1})"]
    return [f"plant, {b}" for b in bands] + [f"empty, {b}" for b in bands]


def load_states(path):
    """src/collect_states.py file -> windows (N, 6, w, w), full states (N, 3, n, n), layout id per state (N,) and the
    monitor observation (N,) (NaN where the file has none)."""
    d = np.load(path)
    n_layouts, steps = d["full"].shape[:2]
    monitor = d["monitor"].reshape(-1).astype(np.float32) if "monitor" in d else np.full(n_layouts * steps, np.nan, np.float32)
    return (d["windows"].reshape(-1, *d["windows"].shape[-3:]).astype(np.float32),
            d["full"].reshape(-1, *d["full"].shape[-3:]).astype(np.float32), np.repeat(d["layout_seed"], steps), monitor)


def predict(model, windows, device, batch=2048, monitor=None):
    """Q-values (N, n_outputs); `monitor` (N,) is the extra input of Q-networks that take the monitor observation."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            m = None if monitor is None else torch.tensor(monitor[i:i + batch], device=device)
            out.append(QWithMonitor(model, m)(torch.tensor(windows[i:i + batch], device=device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", required=True)
    parser.add_argument("--states_dir", default="results/states")
    parser.add_argument("--q_dir", default="results/true_q")
    parser.add_argument("--out_dir", default="results/q_error")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_checks", type=int, default=36)
    parser.add_argument("--pool_checks", type=int, nargs="+", default=None)
    parser.add_argument("--timesteps_per_check", type=float, default=5e5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--monitor_cost", type=float, default=0.2)
    parser.add_argument("--zones", action="store_true",
                        help="also write q_error_by_zone.csv: plant / empty cell x left (monitored) / right / right wall")
    parser.add_argument("--monitor_column_ind", type=int, default=5)
    parser.add_argument("--wall_cols", type=int, default=2)
    parser.add_argument("--mdp_only", action="store_true",
                        help="Q-networks with 6 outputs (no monitor action, e.g. the room monitor): compare with Q*_mdp")
    parser.add_argument("--device", default="mps:0" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    cost = None if args.mdp_only else args.monitor_cost
    tables = solve_layouts(args.states_dir, args.q_dir, args.gamma)  # cached after the first call
    pooled = None
    if args.pool_checks:
        parts = [load_states(os.path.join(args.states_dir, f"seed_{s}", f"checkpoint_{c}.npz"))
                 for s in range(args.n_seeds) for c in args.pool_checks
                 if os.path.exists(os.path.join(args.states_dir, f"seed_{s}", f"checkpoint_{c}.npz"))]
        pooled = tuple(np.concatenate(x) for x in zip(*parts))
        pooled = pooled + (true_q_of_states(pooled[1], pooled[2], tables, cost),)
        print(f"pooled evaluation states: {len(pooled[0])}", flush=True)

    for seed in range(args.n_seeds):
        rows, group_rows, zone_rows = [], [], []
        for check in range(args.n_checks):
            model_path = os.path.join(args.main_dir, f"checkpoints_{check}", f"q_network_{seed}")
            state_path = os.path.join(args.states_dir, f"seed_{seed}", f"checkpoint_{check}.npz")
            if not (os.path.exists(model_path) and os.path.exists(state_path)):
                continue
            model = torch.load(model_path, map_location=device, weights_only=False)
            if pooled is None:
                windows, full, layouts, monitor = load_states(state_path)
                true = true_q_of_states(full, layouts, tables, cost)
            else:
                windows, full, _, monitor, true = pooled
            if uses_monitor(model, torch.zeros(1, 6, 11, 11, device=device)):
                if np.isnan(monitor).any():
                    raise ValueError("Q-network takes the monitor observation but the states file has none (recollect)")
            else:
                monitor = None
            pred, groups = predict(model, windows, device, monitor=monitor), state_groups(full)
            if pred.shape[1] != true.shape[1]:
                raise ValueError(f"Q-network has {pred.shape[1]} outputs, Q* has {true.shape[1]} (use --mdp_only?)")
            meta = {"seed": seed, "checkpoint": check, "timesteps_M": check * args.timesteps_per_check / 1e6}
            rows.append({**meta, **q_metrics(pred, true)})
            if args.zones:
                zones, names = state_zones(full, args.monitor_column_ind, args.wall_cols), zone_names(args.monitor_column_ind, full.shape[-1], args.wall_cols)
                for g, name in enumerate(names):
                    if (zones == g).any():
                        zone_rows.append({**meta, "group": name, "n_states": int((zones == g).sum()),
                                          **q_metrics(pred[zones == g], true[zones == g])})
            for g, name in enumerate(GROUPS):
                if (groups == g).any():
                    group_rows.append({**meta, "group": name, "n_states": int((groups == g).sum()),
                                       **q_metrics(pred[groups == g], true[groups == g])})
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(seed_dir, "q_error.csv"), index=False)
        pd.DataFrame(group_rows).to_csv(os.path.join(seed_dir, "q_error_by_state.csv"), index=False)
        if args.zones:
            pd.DataFrame(zone_rows).to_csv(os.path.join(seed_dir, "q_error_by_zone.csv"), index=False)
        with open(os.path.join(seed_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"seed {seed}: {len(rows)} checkpoints -> {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
