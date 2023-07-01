import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, TypeVar, SupportsFloat

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# E: empty
# S: square (+1)
# T: triangle (-1)
# E: empty
MAPS = {
    "4x8": [
        "EEEEEEEE",
        "ETSTETEE",
        "ETEEETSE",
        "EEEEEEEE",
    ],
}


def _move(row, col, a):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    else:
        raise ValueError('illegal action')
    return (row, col)


class ToyGrid(gym.Env):
    metadata = {}

    def __init__(self, map="4x8", render_mode=None):
        self._map = MAPS[map]
        self._n_rows, self._n_cols = self._map.shape
        self.observation_space = spaces.Discrete(self._n_rows * self._n_cols)
        self.action_space = spaces.Discrete(4)
        self._state = 0

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self._state = self.observation_space._np_random.integers(self._n_rows)
        return self._state, {}

    def step(self, action: ActType):
        shape = (self._n_rows, self._n_cols)
        row, col = np.unravel_index(self._state, shape)
        next_row, next_col = _move(row, col, action)
        self._state = np.unravel_index((next_row, next_col), shape)

        if self._map[next_row, next_col] == 'S':
            self._map[next_row, next_col] == 'E'
            reward = 1
        elif self._map[next_row, next_col] == 'T':
            self._map[next_row, next_col] == 'E'
            reward = -1
        else:
            reward = 0

        terminated = False
        if all([cell == 'E'] for cell in [*[''.join(self._map)]]:
            terminated = True

        return self._state, reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        pass
