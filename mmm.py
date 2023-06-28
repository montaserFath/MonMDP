import gymnasium
from gymnasium import spaces
import numpy as np
from abc import abstractmethod



class MonMDP:
    """
    Generic monitor class.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env, monitor, **kwargs):
        self.env = env
        self.monitor = monitor
        self.

    def reset(self, seed=None, **kwargs):
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        monitor_obs, monitor_info = self.monitor.reset(seed=seed, **kwargs)
        obs = {'env': env_obs, 'monitor': monitor_obs}
        info = env_info | monitor_info
        return obs, info

    def step(self, action):
        env_action = action['env']
        monitor_action = action['monitor']
        env_obs, env_reward, env_terminated, env_truncated, env_info = \
            self.env.step(action['env'])

        monitor_obs, proxy_reward, monitor_cost = \
            self._monitor_step(action, env_reward)

        obs = {'mdp': env_obs, 'monitor': monitor_obs}
        reward = {'mdp': proxy_reward, 'monitor': monitor_cost}
        terminated = env_terminated
        truncated = env_truncated
        info = env_info | {'env_reward': env_reward}

        return obs, reward, terminated, truncated, info



class FullMonitor(gymnasium.Env):
    """
    Monitor always shows the true reward without any cost, regardless of states
    and actions.
    This is equivalent to a classic MDP.

    """
    def __init__(self, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, **kwargs):
        monitor_obs = 0  # default monitor state, always active
        return monitor_obs, {}

    def step(self, action, env_reward):
        monitor_cost = 0.
        proxy_reward = env_reward
        monitor_obs = 0
        return monitor_obs, (proxy_reward, monitor_cost), False, False, {}



class BinaryMonitor(gymnasium.Env):
    """
    Simple monitor where the action is "ask for monitor or not".
    The monitor state is also binary ("monitor is available or not").
    Monitor cost is constant.

    If the agent asks for monitor then it gets to see the true reward at a cost.
    The monitor can then deactive itself randomly.
    If the monitor is not active, the true reward cannot be seen.

    Args:
        monitor_reset_prob (float): probability of the monitor resetting itself,
        monitor_cost (float): cost for monitor request.

    """
    def __init__(self, monitor_cost=0.01, monitor_reset_prob=.5, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(2)
        self.monitor_state = 0  # deactivated
        self.monitor_reset_prob = monitor_reset_prob
        self.monitor_cost = monitor_cost

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.monitor_state = 0
        return self.monitor_state, {}

    def step(self, action, env_reward):
        if action == 1:
            self.monitor_state = 1
            monitor_cost = - self.monitor_cost
        elif action == 0:
            self.monitor_state = 0
            monitor_cost = 0.
        else:
            raise ValueError('illegal monitor action')

        if self.monitor_state == 1:
            proxy_reward = env_reward
        else:
            proxy_reward = np.nan

        if self.monitor_state == 1:
            if np.random.rand() < self.monitor_reset_prob:
                self.monitor_state = 0
        monitor_obs = self.monitor_state

        return monitor_obs, (proxy_reward, monitor_cost), False, False, {}
