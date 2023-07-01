import gymnasium as gym
from gymnasium import spaces
from typing import Any, TypeVar, SupportsFloat

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class ToyWorld(gym.Env):
    metadata = {}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self._state = None

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self._state = 0
        return self._state, {}

    def step(self, action: ActType):
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



from gymnasium.envs.registration import register

def register_envs():
    register(
        id="ToyWorld-v0",
        entry_point="gym_monitor.gym:ToyWorld",
    )
