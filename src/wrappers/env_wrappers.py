"""gym wrappers for MDP & monitor environments"""
import numpy as np
import gymnasium as gym
from minigrid import wrappers as minigrid_wrappers


def wrap_minigrid(env):
    """Wrapper minigrid env"""
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
    TODO: use dict
    """

    def __init__(self, env, n_active_actions: int = 3):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(n_active_actions)

    def action(self, action):
        return action


class TimeStepReward(gym.RewardWrapper):
    """Reward wrapper change goal reward to 0, fire to -50, and timestep penalty to the desired value"""

    def __init__(self, env, goal_reward: float = 0, fire_reward: float = -50, timestep_penalty: float = 0.0):
        super().__init__(env)
        self.timestep_penalty = timestep_penalty
        self.goal_reward = goal_reward
        self.fire_reward = fire_reward

    def reward(self, reward):
        if reward == 1:  # reach the goal
            reward = self.goal_reward
        elif reward == -1:  # steps on a fire
            reward = self.fire_reward
        return reward - self.timestep_penalty


class TabularObservationsWrapper(gym.ObservationWrapper):
    """Convert observations from a list to Tabular"""

    def __init__(self, env, grid_size: tuple = (3, 3)):
        super().__init__(env)
        self.env = env
        self.grid_size = grid_size
        max_obs = int(grid_size[0] * grid_size[1])
        self.observation_space = gym.spaces.Discrete(max_obs)

    def observation(self, obs):
        return np.where(obs == 1)[0]


class WindowViewObs(gym.ObservationWrapper):
    """Observation wrapper get a window view around the agent"""
    def __init__(self, env, grid_size: tuple, window_size: tuple, image_obs: bool = False):
        super().__init__(env)
        self.env = env
        self.window_size = window_size
        self.grid_size = grid_size
        self.image_obs = image_obs
        # TODO: remove 9 with number of objects in the env
        self.observation_space = gym.spaces.Box(0, 255 if image_obs else 9, window_size, dtype=np.int8)

    def observation(self, obs):
        agent_id = 1  # TODO remove hard coded value
        obs = obs.reshape(self.grid_size[0], self.grid_size[1])
        pos_x, pos_y = np.where(obs == agent_id)[0][0], np.where(obs == agent_id)[1][0]
        min_x, max_x = max(0, pos_x - self.window_size[0]), min(self.grid_size[0] - 1, pos_x + self.window_size[0] - 1)
        min_y, max_y = max(0, pos_y - self.window_size[1]), min(self.grid_size[1] - 1, pos_y + self.window_size[0] - 1)
        return obs[min_x: max_x, min_y: max_y]
