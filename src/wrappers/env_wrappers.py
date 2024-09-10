"""gym wrappers for MDP & monitor environments"""
import numpy as np
import gymnasium as gym


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


class StochasticAction(gym.ActionWrapper):
    """Add Stochastic to the action, with some probability take a random action"""

    def __init__(self, env: gym.Env, random_prob: float = 0.0):
        # for discrete actions only for now
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("The action space is not discrete")
        self._random_prob = random_prob
        self.env = env
        if self._random_prob < 0 or random_prob > 1:
            raise ValueError("The random probability should be in [0, 1]")

    def action(self, action):
        if np.random.random() < self._random_prob:
            return self.env.action_space.sample()
        return action


class WindowViewObs(gym.ObservationWrapper):
    """Observation wrapper get a window view around the agent"""

    def __init__(
            self, env, window_size: int,
            agent_channel: int = 0,
            agent_id: int = 1,
            image_obs: bool = False,
            flatten_obs: bool = False,
    ):
        super().__init__(env)
        self.env = env
        old_shape = self.env.observation_space.shape
        if window_size > old_shape[1] or window_size > old_shape[2]:
            raise ValueError("The window size should smaller than half of the grid size")
        self.window_size = window_size
        self.agent_channel = agent_channel
        self.agent_id = agent_id
        self.image_obs = image_obs
        self.flatten_obs = flatten_obs

        high = 255 if image_obs else int(env.observation_space.high_repr)
        low = 0 if image_obs else int(env.observation_space.low_repr)

        if len(self.env.observation_space.shape) != 3:
            raise ValueError("Parent env should have 3 dimensions CxHxW")
        self.observation_space = gym.spaces.Box(low, high, (old_shape[0], window_size, window_size), dtype=np.int8)

    def observation(self, obs):
        # window_obs = np.zeros((self.window_size[0], self.window_size[1], self.env.observation_space.shape))
        last_x, last_y = self.env.observation_space.shape[1], self.env.observation_space.shape[2]
        pos_x = np.where(obs[self.agent_channel, :, :] == self.agent_id)[0][0]
        pos_y = np.where(obs[self.agent_channel, :, :] == self.agent_id)[1][0]
        min_x, max_x = max(0, pos_x - self.window_size // 2), min(last_x, pos_x + self.window_size // 2 + 1)
        min_y, max_y = max(0, pos_y - self.window_size // 2), min(last_y, pos_y + self.window_size // 2 + 1)
        window_obs = obs[:, min_x:max_x, min_y:max_y]
        if window_obs.shape[1] != self.window_size or window_obs.shape[2] != self.window_size:
            raise ValueError("mismatch shape of observation in window wrapper")
        return window_obs.reshape(-1) if self.flatten_obs else window_obs


class ChannelsObs(gym.ObservationWrapper):
    def __init__(self, env, grid_size: tuple, n_objects: int):
        super().__init__(env)
        self.env = env
        self.n_objects = n_objects
        self.grid_size = grid_size
        self.observation_space = gym.spaces.Box(
            0,
            self.n_objects,
            shape=(self.n_objects, self.grid_size[0], self.grid_size[1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        """
        AGENT = channel 0
        GOAL = channel 1
        FIRE = channel 2
        BUTTON = channel 3
        WALL = channel 4
        """
        agent_position = list(self.env.get_agent_pos())
        grid = self.env.get_grid()
        observation = np.zeros((self.n_objects, self.grid_size[0], self.grid_size[1]))
        for obj_id in range(1, self.n_objects + 1):
            if obj_id == 1:  # agent
                observation[0, agent_position[0], agent_position[1]] = 1
            else:
                idx = np.argwhere(grid == obj_id)
                observation[obj_id - 1, idx[:, 0], idx[:, 1]] = 1
        return observation


class WallObs(gym.ObservationWrapper):
    def __init__(self, env, grid_size: tuple, n_walls: int):
        super().__init__(env)
        self.env = env
        if n_walls > (grid_size[0] // 2):
            raise ValueError("number of walls should smaller than half of the grid size")
        self.n_walls = n_walls
        self.grid_size = grid_size
        new_shape = list(self.env.observation_space.shape)
        new_shape[0] += 1
        new_shape[1] += 2 * self.n_walls
        new_shape[2] += 2 * self.n_walls
        self.shape = tuple(new_shape)
        self.observation_space = gym.spaces.Box(0, self.shape[0], shape=self.shape, dtype=np.uint8)

    def observation(self, obs):
        new_obs = np.zeros(self.shape)
        # copy obs
        new_obs[:-1, self.n_walls : -self.n_walls, self.n_walls : -self.n_walls] = obs.copy()

        # walls channel
        new_obs[-1, : self.n_walls, :] = 1
        new_obs[-1, :, : self.n_walls] = 1
        new_obs[-1, -self.n_walls :, :] = 1
        new_obs[-1, :, -self.n_walls :] = 1
        return new_obs

    def get_n_walls(self):
        return self.n_walls
