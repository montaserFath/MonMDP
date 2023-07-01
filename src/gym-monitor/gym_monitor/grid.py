from contextlib import closing
import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils
from typing import Any, TypeVar, SupportsFloat, Optional
from io import StringIO

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


def _move(row, col, a, nrow, ncol):
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
    """
    Gridworld where the agent has to rewards while avoiding penalties.
    The position of rewards and penalties is defined by a map passed as text.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    The action shape is `(1,)` in the range `{0, rows*cols}` indicating the
    cell where the agent is at.

    ## Starting State
    The game starts with the agent at location [0, any].

    ## Rewards
    - Walk over reward: +1
    - Walk over penalty: -1
    - Else: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. All rewards have been collected.

    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 200 for the 4x8 environment.

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('ToyGrid-v1', map="4x8")
    ```

    `map="4x8"`: Uses the pre-defined 4x8 map.

    Specify a custom map.
    ```
        map=["SEEE", "ETET", "EEET", "TEES"].
    ```

    """
    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, map="4x8", **kwargs):
        self.render_mode = render_mode
        self._map = np.asarray(MAPS[map], dtype="c")
        self._n_rows, self._n_cols = self._map.shape
        self.observation_space = spaces.Discrete(self._n_rows * self._n_cols)
        self.action_space = spaces.Discrete(4)
        self._state = 0
        self._last_action = None

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self._state = self.np_random.integers(self._n_rows)
        self._last_action = None
        return self._state, {}

    def step(self, action: ActType):
        shape = (self._n_rows, self._n_cols)
        row, col = np.unravel_index(self._state, shape)
        next_row, next_col = _move(row, col, action, self._n_rows, self._n_cols)
        self._state = np.ravel_multi_index((next_row, next_col), shape)

        if self._map[next_row, next_col] == b'S':
            self._map[next_row, next_col] == b'E'
            reward = 1
        elif self._map[next_row, next_col] == b'T':
            self._map[next_row, next_col] == b'E'
            reward = -1
        else:
            reward = 0

        if (self._map == b'E').all():
            terminated = True
        else:
            terminated = False

        self._last_action = action

        return self._state, reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "ansi":
            return self._render_text()

    def _render_text(self):
        map = self._map.tolist()
        outfile = StringIO()

        row, col = self._state // self._n_cols, self._state % self._n_cols
        map = [[c.decode("utf-8") for c in line] for line in map]
        map[row][col] = utils.colorize(map[row][col], "red", highlight=True)
        if self._last_action is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self._last_action]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in map) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        pass
