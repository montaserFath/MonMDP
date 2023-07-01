import gymnasium as gym
from gymnasium import spaces
from typing import Any, TypeVar, SupportsFloat

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

# E: empty
# S: square
# T: triangle
# E: empty
MAPS = {
    "4x8": [
        "EEEEEEEE",
        "ETSTETEE",
        "ETEEETSE",
        "EEEEEEEE",
    ],
}


class MonitoredWorld(gym.Env):
    metadata = {}

    def __init__(self, map="4x8", render_mode=None):
        self._map = MAPS[map]
        n_rows, n_cols = self._map.shape
        self.observation_space = spaces.Discrete(n_rows * n_cols)
        self.action_space = spaces.Discrete(4)

    def reset(
        self, seed: int | None = None, **kwargs
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, **kwargs)
        return 0, {}

    def step(self,
        action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        trunc = False
        
        if self._state == 0:
            term = False
            if action == 0:
                reward = -10
                self._state = 2
            else:
                reward = 0
                self._state = 1

        elif self._state == 1:
            term = True
            reward = 1

        else:
            term = True
            reward = 2

        return self._state, reward, term, trunc, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        pass
