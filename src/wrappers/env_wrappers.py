import numpy as np
import gymnasium as gym
from minigrid import wrappers as minigrid_wrappers


def wrap_minigrid(env):
    env = minigrid_wrappers.FullyObsWrapper(env)
    env = minigrid_wrappers.ImgObsWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    if "Lava" in env.unwrapped.spec.id:
        env = minigrid_wrappers.NoDeath(env)
    env = minigrid_wrappers.ReseedWrapper(env, seeds=(0,))
    return env


class ActiveActionsWrapper(gym.ActionWrapper):
    """
    Use active actions only
    TODO(Monta): use dict
    """

    def __init__(self, env, n_active_actions: int = 3):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n_active_actions)

    def action(self, action):
        return action


class TimeStepReward(gym.RewardWrapper):
    """Reward wrapper to decay with timestimes"""

    def __init__(self, env, decay_rate: float = 0.05):
        super().__init__(env)
        self.decay_rate = decay_rate

    def reward(self, reward):
        return reward - self.decay_rate


class TabularObservationsWrapper(gym.ObservationWrapper):
    def __init__(self, env, grid_size: tuple = (3, 3)):
        super().__init__(env)
        self.env = env
        self.grid_size = grid_size
        max_obs = grid_size[0] * grid_size[1] - 1
        self.observation_space = gym.spaces.Box(0, max_obs, (1,), dtype="uint8")

    def observation(self, obs):
        return np.where(obs == 1)[0]


class StableBaselinesWrapper(gym.Wrapper):
    def __int__(self, env):
        super.__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, info

    def reset(self, seed: int | None = None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        return obs, info
