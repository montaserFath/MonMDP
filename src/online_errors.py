"""Log the reward-model error and the Q-network TD error during training, in long format (one row per group).

Every `frequency` env steps a batch of `n_samples` transitions is drawn from the replay buffer (the states the agent
actually trains on) and two CSVs are appended to the experiment's log dir:

  online_reward_error_<seed>.csv  one row per (timestep, source, object, dryness, action[, observed, current_table])
      reward model prediction vs the true reward. Columns: n, true_mean, pred_mean, bias (pred - true), mae, mse.
      source = "oracle": the env's own reward function under the *current* reward table, for all 6 actions of every
          sampled state (observed and current_table are empty).
      source = "buffer": the true reward the env gave for the action that was actually taken in each sampled
          transition (stored next to the proxy reward, never used for training), for that action only.
          observed: True if the agent saw that reward, False if it was hidden (NaN) - the Mon-MDP generalisation
          split. current_table: True if the transition was collected under the reward table that is active now, False
          if under an earlier one (then `true` is that earlier table's reward; always True for a stationary env).
  online_td_error_<seed>.csv      one row per (timestep, object, dryness, mdp_action, monitor_action, terminal)
      TD error r + gamma * max Q_target(s', .) - Q(s, a) as the critic forms it for training (reward-model reward).
      Columns: n, td_mean (signed), td_abs_mean, td_sq_mean, q_mean.

Both carry timestep, episode, task_idx (index of the active reward table, -1 for a stationary env) and seed. `object`
is decoded from the centre pixel of the observation window: empty / plant / cactus / more_plant.
To aggregate over any columns, weight by n:  mse = sum(n * mse) / sum(n),  td_abs_mean likewise.

Logging does not change training: the global python / numpy / torch RNG states are restored after sampling, and
nothing is trained or updated.
"""
import csv
import os
import random

import numpy as np
import torch

from src.critic import MonRoomCNN
from src.reward_error import ACTION_NAMES, decode_state, env_reward

META = ["timestep", "episode", "task_idx", "seed"]
REWARD_FIELDS = META + ["source", "object", "dryness", "action", "observed", "current_table", "n", "true_mean",
                        "pred_mean", "bias", "mae", "mse"]
TD_FIELDS = META + ["object", "dryness", "mdp_action", "monitor_action", "terminal", "n", "td_mean", "td_abs_mean",
                    "td_sq_mean", "q_mean"]


def decode_object(vec) -> tuple:
    """Centre pixel [agent, R, G, B, dryness, wall] -> (object, dryness); the extra plant types of the more-plants env
    (other colours) are reported as 'more_plant'."""
    try:
        return decode_state(vec)
    except ValueError:
        agent, red, green, blue, dryness, wall = vec
        if agent == 1 and wall == 0 and (red > 0 or green > 0 or blue > 0):
            return "more_plant", float(dryness)
        raise


def decode_batch(mdp_obs: torch.Tensor):
    """Observation windows (N, C, H, W) -> unique (object, dryness) list and, per sample, the index into that list."""
    centre = mdp_obs[:, :, mdp_obs.shape[-2] // 2, mdp_obs.shape[-1] // 2].detach().cpu().numpy().round(4)
    unique, inverse = np.unique(centre, axis=0, return_inverse=True)
    return [decode_object(v) for v in unique], inverse.reshape(-1)


class OnlineErrorLogger:
    """Appends reward-model error and TD error rows to CSVs every `frequency` env steps."""

    def __init__(self, log_dir: str, seed: int, frequency: int, n_samples: int = 2048):
        self.log_dir, self.seed = log_dir, seed
        self.frequency, self.n_samples = int(frequency), int(n_samples)
        self.reward_path = os.path.join(log_dir, f"online_reward_error_{seed}.csv")
        self.td_path = os.path.join(log_dir, f"online_td_error_{seed}.csv")
        self._next = 0
        self._oracle_env = None

    # ------------------------------------------------------------------ schedule
    def maybe_log(self, timestep: int, episode: int, buffer, critic, env) -> bool:
        """Log if `timestep` reached the next log step and the buffer holds enough transitions."""
        if self.frequency <= 0 or timestep < self._next or len(buffer) <= self.n_samples:
            return False
        self._next = (timestep // self.frequency + 1) * self.frequency
        self.log(timestep, episode, buffer, critic, env)
        return True

    def log(self, timestep: int, episode: int, buffer, critic, env) -> None:
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        try:
            batch = buffer.process_batch(buffer.sample(self.n_samples), device=critic.get_device())
        finally:
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
            torch.set_rng_state(rng_state[2])
        meta = {"timestep": int(timestep), "episode": int(episode), "task_idx": getattr(env, "task_idx", -1),
                "seed": self.seed}
        objects, inverse = decode_batch(batch["mdp_obs"])
        if getattr(critic, "_r_model", None) is not None:
            self._append(self.reward_path, REWARD_FIELDS, self.reward_rows(batch, objects, inverse, critic, env, meta))
        if hasattr(critic, "td_error"):
            self._append(self.td_path, TD_FIELDS, self.td_rows(batch, objects, inverse, critic, meta))

    # ------------------------------------------------------------------ reward error
    def oracle(self, env, objects) -> np.ndarray:
        """(len(objects), 6) env reward for each decoded object and action, under the env's current reward table."""
        base = env.unwrapped
        if self._oracle_env is None:  # separate small env: `env_reward` overwrites the env's grid and positions
            self._oracle_env = type(base)(
                grid_size=(5, 5), n_plants=1, plants_dryness_prob=0.0, dry_difference=base.dry_difference,
                max_episode_steps=1, add_new_plants=base.add_new_plants, add_more_plants=base.add_more_plants,
            )
        self._oracle_env.set_new_reward_function(dict(base.reward_table))
        # a more-plant is rewarded like a cactus (key new_plant)
        return np.array([[env_reward(self._oracle_env, "cactus" if kind == "more_plant" else kind, dryness, a)
                          for a in range(len(ACTION_NAMES))] for kind, dryness in objects])

    def reward_rows(self, batch, objects, inverse, critic, env, meta) -> list:
        with torch.no_grad():
            pred = critic._r_model(batch["mdp_obs"]).detach().cpu().numpy()  # (N, 6), as the critic calls it
        true = self.oracle(env, objects)[inverse]
        err = pred - true
        rows = []
        for i, (kind, dryness) in enumerate(objects):
            mask = inverse == i
            for a, name in enumerate(ACTION_NAMES):
                rows.append({**meta, "source": "oracle", "object": kind, "dryness": dryness, "action": name,
                             **self._stats(pred[mask, a], true[mask, a])})
        return rows + self.buffer_reward_rows(batch, objects, inverse, pred, env, meta)

    @staticmethod
    def _stats(pred: np.ndarray, true: np.ndarray) -> dict:
        err = pred - true
        return {"n": len(err), "true_mean": float(true.mean()), "pred_mean": float(pred.mean()),
                "bias": float(err.mean()), "mae": float(np.abs(err).mean()), "mse": float((err ** 2).mean())}

    def buffer_reward_rows(self, batch, objects, inverse, pred, env, meta) -> list:
        """Error against the true reward stored with each sampled transition, for the action that was taken, split by
        whether the agent observed that reward and whether it was collected under the reward table active now."""
        true = batch["true_reward"].squeeze(1).cpu().numpy()
        keep = ~np.isnan(true)  # transitions stored without a true reward (older code) are skipped
        if not keep.any():
            return []
        action = batch["mdp_action"].squeeze(1).cpu().numpy().astype(int)
        observed = ~np.isnan(batch["mdp_reward"].squeeze(1).cpu().numpy())
        current = batch["change_idx"].squeeze(1).cpu().numpy() == getattr(env, "n_changes", -1)
        predicted = pred[np.arange(len(pred)), action]  # the model's output for the action that was taken
        keys = np.stack([inverse, action, observed.astype(int), current.astype(int)], axis=1)
        rows = []
        for key in np.unique(keys[keep], axis=0):
            mask = (keys == key).all(1) & keep
            kind, dryness = objects[key[0]]
            rows.append({**meta, "source": "buffer", "object": kind, "dryness": dryness, "action": ACTION_NAMES[key[1]],
                         "observed": bool(key[2]), "current_table": bool(key[3]),
                         **self._stats(predicted[mask], true[mask])})
        return rows

    # ------------------------------------------------------------------ TD error
    def td_rows(self, batch, objects, inverse, critic, meta) -> list:
        out = critic.td_error(batch, monitor_state=isinstance(critic, MonRoomCNN))
        rows_idx = out["rows"].cpu().numpy()
        td, q = out["td"].cpu().numpy(), out["q"].cpu().numpy()
        obj = inverse[rows_idx]
        mdp_action = batch["mdp_action"].squeeze(1).cpu().numpy()[rows_idx]
        mon_action = batch["mon_action"].squeeze(1).cpu().numpy()[rows_idx]
        terminal = ~batch["non_final_mask"].cpu().numpy()[rows_idx]
        keys = np.stack([obj, mdp_action, mon_action, terminal.astype(int)], axis=1)
        rows = []
        for key in np.unique(keys, axis=0):
            mask = (keys == key).all(1)
            kind, dryness = objects[key[0]]
            rows.append({**meta, "object": kind, "dryness": dryness, "mdp_action": ACTION_NAMES[key[1]],
                         "monitor_action": int(key[2]), "terminal": bool(key[3]), "n": int(mask.sum()),
                         "td_mean": float(td[mask].mean()), "td_abs_mean": float(np.abs(td[mask]).mean()),
                         "td_sq_mean": float((td[mask] ** 2).mean()), "q_mean": float(q[mask].mean())})
        return rows

    @staticmethod
    def _append(path: str, fields: list, rows: list) -> None:
        new_file = not os.path.exists(path)
        if not new_file:  # never mix column layouts in one file (e.g. a CSV written by an older version)
            with open(path, newline="") as f:
                header = next(csv.reader(f), [])
            if header != fields:
                raise ValueError(f"{path} has columns {header}, expected {fields}: move or delete the old file")
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
