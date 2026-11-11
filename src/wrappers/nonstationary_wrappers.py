"""Non-stationary wrappers: change the reward of the plant watering env every fixed number of env timesteps."""
from typing import Optional, Sequence

import gymnasium
import numpy as np

ENV_KEYS = ("dry_plant", "half_dry_plant", "empty_cell", "wet_plant")
TABLE_KEYS = ENV_KEYS + ("monitor",)


def validate_table(table: dict) -> None:
    """Check one reward table: dry >= half-dry > 0 > floor(empty_cell) > wet and floor <= monitor <= 0."""
    missing = set(TABLE_KEYS) - set(table)
    unknown = set(table) - set(TABLE_KEYS)
    if missing or unknown:
        raise ValueError(f"Reward table keys: missing {sorted(missing)}, unknown {sorted(unknown)}")
    dry, half, floor, wet, mon = (table[k] for k in ("dry_plant", "half_dry_plant", "empty_cell", "wet_plant", "monitor"))
    if not (dry >= half > 0 > floor > wet):
        raise ValueError(f"Need dry >= half_dry > 0 > empty_cell > wet, got {dict(table)}")
    if not (floor <= mon <= 0):
        raise ValueError(f"Need empty_cell <= monitor <= 0, got {dict(table)}")


class RewardSchedule(gymnasium.Wrapper):
    """Switch the reward table of the plant watering env every `period_timesteps` env steps (no episode notion).

    Wrap this *outside* the monitor wrapper. The agent is not told about the change, the task index is only
    available as `task_idx` / `info["task_idx"]` for logging.

    Args:
        env: the (monitor wrapped) env.
        tables: list of dicts with keys dry_plant, half_dry_plant, empty_cell, wet_plant, monitor. `monitor` is the
            monitor reward (<= 0), i.e. the monitor cost is `-monitor`.
        period_timesteps: number of env steps between two changes.
        order: "cycle" (0, 1, ..., n-1, 0, ...) or "random" (uniform among the other tables).
        vary_monitor_cost: if False (default) the monitor cost is left untouched (fixed, e.g. 0.2 from the monitor
            config) and the `monitor` column is ignored. If True the monitor cost follows the table.
        seed: seed of the schedule's own RNG (never touches the global numpy RNG).
    """

    def __init__(
        self,
        env,
        tables: Sequence[dict],
        period_timesteps: int = 2000,
        order: str = "cycle",
        vary_monitor_cost: bool = False,
        seed: int = 0,
    ):
        super().__init__(env)
        if order not in ("cycle", "random"):
            raise ValueError("order must be 'cycle' or 'random'")
        if period_timesteps < 1:
            raise ValueError("period_timesteps must be >= 1")
        self.tables = [{k: float(v) for k, v in t.items()} for t in tables]
        if not self.tables:
            raise ValueError("tables is empty")
        for table in self.tables:
            validate_table(table)
        self.period_timesteps = int(period_timesteps)
        self.order = order
        self.vary_monitor_cost = vary_monitor_cost
        self._rng = np.random.default_rng(seed)
        self._paused = False
        self._steps = 0
        self.n_changes = 0
        self.task_idx = 0
        self._apply(0)

    def set_schedule_paused(self, paused: bool) -> None:
        """While paused, steps do not advance the clock (use during evaluation)."""
        self._paused = bool(paused)

    def _apply(self, idx: int) -> None:
        self.task_idx = idx
        table = self.tables[idx]
        self.env.unwrapped.set_new_reward_function({k: table[k] for k in ENV_KEYS})
        if self.vary_monitor_cost:
            if not hasattr(self.env, "monitor_cost"):
                raise ValueError("vary_monitor_cost=True but the wrapped env has no `monitor_cost`")
            self.env.monitor_cost = -table["monitor"]

    def _next_idx(self) -> int:
        n = len(self.tables)
        if self.order == "cycle" or n == 1:
            return (self.task_idx + 1) % n
        others = [i for i in range(n) if i != self.task_idx]
        return int(self._rng.choice(others))

    def step(self, action):
        if not self._paused:
            if self._steps > 0 and self._steps % self.period_timesteps == 0:
                self.n_changes += 1
                self._apply(self._next_idx())
            self._steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info | {"task_idx": self.task_idx, "n_changes": self.n_changes}
