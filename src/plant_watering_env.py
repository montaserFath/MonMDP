import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame


class PlantWateringEnv(gym.Env):
    def __init__(
            self, grid_size: (int, int), n_plants: int, agent_start_pos: int, seed: int, max_episode_steps: int, render_mode: str = 'rgb_array'
    ):
        metadata = {
            "render_modes": ["human", "rgb_array", "ansi"],
            "render_fps": 4,
        }
        self.n_raws, self._n_columns = grid_size
        self.n_plants = n_plants
        self.agent_start_pos = agent_start_pos
        self.seed = seed
        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box()
        self.reward_range = (0, 1)

    def step(self, action):
        NotImplemented

    def reset(self):
        NotImplemented

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        NotImplemented

    def close(self):
        NotImplemented

    def seed(self, seed):
        NotImplemented

    def move(self):
        NotImplemented