"""Roll out each saved Q-network checkpoint and record, at every step, the FULL environment state next to the
PARTIAL observation (the window) the agent actually sees.

Why: the true value of a state (Q*) depends on the whole grid (all plant positions and dryness), but the Q-network only
sees an 11x11 window. To compare the two, both must be recorded for the same state. The env already returns the full
grid in info["grid"]; this script pairs it with obs["mdp"].

The env stack, joint-action decoding (mdp = idx % 6, monitor = idx // 6) and epsilon-greedy choice mirror
run_plants_env.py / src/actor.py. Layouts are fixed by seed (layout_seed + k), so every checkpoint is evaluated on the
same plant layouts and Q* only has to be computed once per layout.

Usage:
  python src/collect_states.py --main_dir <dir with checkpoints_*/> --out_dir results/states [--seeds 0 1 --checks 1 10 30]
Writes <out_dir>/seed_<s>/checkpoint_<c>.npz with arrays
  full (L,T,3,H,W) agent/plant/dryness grid, windows (L,T,6,w,w), actions (L,T,2) [mdp, monitor],
  mdp_reward (L,T) true env reward, agent_pos (L,T,2), monitor (L,T) monitor observation, layout_seed (L,)
where L = number of layouts and T = steps per episode; state t is the state the action t was chosen in.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.checkpoints.evaluate_rl_checkpoints import QWithMonitor, uses_monitor  # noqa: E402
from src.reward_error import load_env_class  # noqa: E402
from src.wrappers.env_wrappers import MultiChannel, WallObs, WindowViewObs  # noqa: E402
from src.wrappers.monitor_wrappers import BinaryMonitor, RoomMonitor  # noqa: E402

N_MDP_ACTIONS = 6


def build_env(grid=10, n_plants=8, dryness_prob=0.05, dry_difference=0.5, max_steps=100, n_walls=5, window=11,
              monitor_cost=0.2, normalize_obs=True, start_pos=(0, 0), monitor="binary", monitor_column_ind=5):
    """Same wrapper stack as run_plants_env.py. monitor='room': left columns < monitor_column_ind are monitored (no
    cost) and the agent starts in a random cell, as in run_plants_env.py for RoomMonitor."""
    if monitor == "room":
        start_pos = None
    env = load_env_class()(grid_size=(grid, grid), n_plants=n_plants, plants_dryness_prob=dryness_prob,
                           dry_difference=dry_difference, agent_start_pos=None if start_pos is None else list(start_pos), max_episode_steps=max_steps,
                           add_new_plants=False, add_more_plants=False)
    env = WallObs(env, grid_size=(grid, grid), n_walls=n_walls)
    env = WindowViewObs(env, window_size=window)
    env = MultiChannel(env, normalize_obs=normalize_obs)
    if monitor == "room":
        return RoomMonitor(env, full_monitor=False, monitor_cost=0.0, monitor_column_ind=monitor_column_ind)
    return BinaryMonitor(env, full_monitor=False, monitor_cost=monitor_cost, monitor_reset_prob=0.0, init_monitor_state=0)


def decode_action(idx: int) -> dict:
    """Joint action index -> {'mdp', 'monitor'} (src/actor.py ind_to_action)."""
    return {"mdp": int(idx % N_MDP_ACTIONS), "monitor": int(idx // N_MDP_ACTIONS)}


def rollout(q_net, envs, layout_seeds, max_steps, eps, rng, device) -> dict:
    """Run one episode on every layout in lockstep (the Q-network is called once per step on the whole batch)."""
    n = len(envs)
    windows, grids, mon = [], [], []
    for env, seed in zip(envs, layout_seeds):
        np.random.seed(seed)  # the env draws the layout from the global numpy RNG
        obs, info = env.reset()
        windows.append(obs["mdp"]), grids.append(np.array(info["grid"])), mon.append(obs["monitor"])
    first = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
    takes_monitor = uses_monitor(q_net, first)
    with torch.no_grad():  # 12 outputs: 6 mdp x 2 monitor actions; 6 outputs: no monitor action (room monitor)
        n_monitor_actions = QWithMonitor(q_net, torch.tensor(mon, dtype=torch.float32, device=device) if takes_monitor
                                         else None)(first).shape[1] // N_MDP_ACTIONS

    rec = {k: [] for k in ("full", "windows", "actions", "mdp_reward", "agent_pos", "monitor")}
    for _ in range(max_steps):
        x = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
        m = torch.tensor(mon, dtype=torch.float32, device=device) if takes_monitor else None
        with torch.no_grad():
            greedy = QWithMonitor(q_net, m)(x).argmax(1).cpu().numpy()
        actions = [decode_action(a) for a in greedy]
        for i in np.flatnonzero(rng.random(n) < eps):  # epsilon-greedy over mdp and monitor actions independently
            actions[i] = {"mdp": int(rng.integers(N_MDP_ACTIONS)), "monitor": int(rng.integers(n_monitor_actions))}
        rec["full"].append(np.stack(grids)), rec["windows"].append(np.stack(windows)), rec["monitor"].append(list(mon))
        rec["actions"].append([[a["mdp"], a["monitor"]] for a in actions])
        rewards, positions = [], []
        for i, env in enumerate(envs):
            obs, _, _, _, info = env.step(actions[i])
            windows[i], grids[i], mon[i] = obs["mdp"], np.array(info["grid"]), obs["monitor"]
            rewards.append(info["mdp_reward"])
            positions.append(info["previous_agent_pos"])  # position the action was taken from
        rec["mdp_reward"].append(rewards), rec["agent_pos"].append(positions)
    out = {k: np.stack(v, axis=1) for k, v in rec.items()}  # (L, T, ...)
    out["full"], out["windows"] = out["full"].astype(np.float16), out["windows"].astype(np.float16)
    out["actions"], out["agent_pos"] = out["actions"].astype(np.int8), out["agent_pos"].astype(np.int8)
    out["mdp_reward"] = out["mdp_reward"].astype(np.float32)
    out["monitor"] = out["monitor"].astype(np.int8)  # monitor observation the network received at each state
    out["layout_seed"] = np.array(layout_seeds)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", required=True)
    parser.add_argument("--out_dir", default="results/states")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--checks", type=int, nargs="+", default=list(range(36)))
    parser.add_argument("--n_layouts", type=int, default=10)
    parser.add_argument("--layout_seed", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--n_plants", type=int, default=8)
    parser.add_argument("--dryness_prob", type=float, default=0.05)
    parser.add_argument("--dry_difference", type=float, default=0.5)
    parser.add_argument("--n_walls", type=int, default=5)
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--monitor_cost", type=float, default=0.2)
    parser.add_argument("--monitor", choices=["binary", "room"], default="binary")
    parser.add_argument("--monitor_column_ind", type=int, default=5)
    parser.add_argument("--no_normalize_obs", action="store_true",
                        help="raw plant colour (1,1,0) instead of the (0.5,0.5,0) used by run_plants_env.py")
    parser.add_argument("--device", default="mps:0" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    envs = [build_env(args.grid, args.n_plants, args.dryness_prob, args.dry_difference, args.max_steps, args.n_walls,
                      args.window, args.monitor_cost, normalize_obs=not args.no_normalize_obs, monitor=args.monitor,
                      monitor_column_ind=args.monitor_column_ind)
            for _ in range(args.n_layouts)]
    layout_seeds = [args.layout_seed + k for k in range(args.n_layouts)]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for seed in args.seeds:
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        done = 0
        for check in args.checks:
            path = os.path.join(args.main_dir, f"checkpoints_{check}", f"q_network_{seed}")
            if not os.path.exists(path):
                continue  # this seed's run ended before this checkpoint
            q_net = torch.load(path, map_location=device, weights_only=False).eval()
            rng = np.random.default_rng(seed * 1000 + check)
            data = rollout(q_net, envs, layout_seeds, args.max_steps, args.eps, rng, device)
            np.savez_compressed(os.path.join(seed_dir, f"checkpoint_{check}.npz"), **data)
            done += 1
        print(f"seed {seed}: {done} checkpoints -> {seed_dir}", flush=True)


if __name__ == "__main__":
    main()
