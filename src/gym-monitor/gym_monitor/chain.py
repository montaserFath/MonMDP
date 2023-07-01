import gymnasium as gym
from gymnasium import spaces
from typing import Any, TypeVar, SupportsFloat, Optional

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


class ToyChain(gym.Env):
    """
    Toy 3-state chain world: the agent starts in state A and can go to eithe
    state B (reward -10) or state C (reward 0). From both state B and C, the
    agent goes back to state A. Going back from state B gives a reward of 2,
    while going back from state C gives a reward of 1.

    The optimal policy always goes A -> C -> A -> C ...

           ---> B ---> A
      -10 /        2
         /
    A ---
         \
        0 \        1
           ---> C ---> A

    """
    metadata = {
        "render_modes" : [],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self._state = None

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed)
        self._state = 0
        return self._state, {}

    def step(self, action: ActType):
        if self._state == 0:
            term = False
            if action == 0:
                reward = -10
                self._state = 2
            elif action == 1:
                reward = 0
                self._state = 1
            else:
                raise ValueError('illegal action')
        elif self._state == 1:
            self._state = 0
            reward = 1
        else:
            self._state = 0
            reward = 2

        return self._state, reward, False, False, {}

    def render(self):
        pass

    def close(self):
        pass
