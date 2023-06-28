import gymnasium
from gymnasium import spaces
import numpy as np
from abc import abstractmethod



class Monitor(gymnasium.Wrapper):
    """
    Generic monitor class.

    Args:
        env (gymnasium.Env): the Gymnasium environment.

    """
    @abstractmethod
    def _monitor_step(self, action, mdp_reward):
        pass

    def step(self, action):
        mdp_obs, mdp_reward, mdp_terminated, mdp_truncated, mdp_info = \
            self.env.step(action['mdp'])

        monitor_obs, proxy_reward, monitor_cost = \
            self._monitor_step(action, mdp_reward)

        obs = {'mdp': mdp_obs, 'monitor': monitor_obs}
        reward = {'mdp': proxy_reward, 'monitor': monitor_cost}
        terminated = mdp_terminated
        truncated = mdp_truncated
        info = mdp_info | {'mdp_reward': mdp_reward}

        return obs, reward, terminated, truncated, info



class FullMonitor(Monitor):
    """
    Monitor always shows the true reward without any cost, regardless of states
    and actions.
    This is equivalent to a classic MDP.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_reset_prob (float): probability of the monitor resetting itself.

    """
    def __init__(self, env, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            'mdp': env.action_space,
            'monitor': spaces.Discrete(1),
        })
        self.observation_space = spaces.Dict({
            'mdp': env.observation_space,
            'monitor': spaces.Discrete(1),
        })

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        monitor_obs = 0  # default monitor state, always active
        return {'mdp': mdp_obs, 'monitor': monitor_obs}, mdp_info

    def _monitor_step(self, action, mdp_reward):
        monitor_cost = 0.
        proxy_reward = mdp_reward
        monitor_obs = 0
        return monitor_obs, proxy_reward, monitor_cost



class BinaryMonitor(Monitor):
    """
    Simple monitor where the action is "ask for monitor or not".
    The monitor state is also binary ("monitor is available or not").
    Monitor cost is constant.

    If the agent asks for monitor then it gets to see the true reward at a cost.
    The monitor can then deactive itself randomly.
    If the monitor is not active, the true reward cannot be seen.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_reset_prob (float): probability of the monitor resetting itself,
        monitor_cost (float): cost for monitor request.

    """
    def __init__(self, env, monitor_cost=0.01, monitor_reset_prob=.5, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            'mdp': env.action_space,
            'monitor': spaces.Discrete(2),
        })
        self.observation_space = spaces.Dict({
            'mdp': env.observation_space,
            'monitor': spaces.Discrete(2),
        })
        self.monitor_state = 0  # deactivated
        self.monitor_reset_prob = monitor_reset_prob
        self.monitor_cost = monitor_cost

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 0
        return {'mdp': mdp_obs, 'monitor': self.monitor_state}, mdp_info

    def _monitor_step(self, action, mdp_reward):
        if action['monitor'] == 1:
            self.monitor_state = 1
            monitor_cost = - self.monitor_cost
        elif action['monitor'] == 0:
            self.monitor_state = 0
            monitor_cost = 0.
        else:
            raise ValueError('illegal monitor action')

        if self.monitor_state == 1:
            proxy_reward = mdp_reward
        else:
            proxy_reward = np.nan

        if self.monitor_state == 1:
            if np.random.rand() < self.monitor_reset_prob:
                self.monitor_state = 0
        monitor_obs = self.monitor_state

        return monitor_obs, proxy_reward, monitor_cost
