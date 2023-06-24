import gymnasium
from gymnasium import spaces
import numpy as np


class RandomMonitor(gymnasium.Wrapper):
    """
    Simple monitor where the action is "ask for monitor or not".
    The monitor state is also binary ("monitor is available or not").
    The monitor state transition is random.
    Monitor cost is constant.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            'mdp': env.action_space,
            'monitor': spaces.Discrete(2),
        })
        self.observation_space = spaces.Dict({
            'mdp': env.observation_space,
            'monitor': spaces.Discrete(2),
        })
        self.monitor_state = 0 # deactivated

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 0
        return {'mdp': mdp_obs, 'monitor': self.monitor_state}, mdp_info

    def step(self, action):
        mdp_action = action['mdp']
        monitor_action = action['monitor']

        mdp_obs, mdp_reward, mdp_terminated, mdp_truncated, mdp_info = \
            self.env.step(mdp_action)

        if monitor_action == 1: # ask for monitor
            self.monitor_state = 1 # activate monitor
            monitor_reward = -0.1 # pay cost
        else:
            monitor_reward = 0.

        if self.monitor_state == 1: # if monitor is active
            proxy_reward = mdp_reward # get proxy reward
        else: # otherwise get undefined
            proxy_reward = np.nan

        # if monitor is active, there is a 50% chance it turns off
        if self.monitor_state == 1:
            self.monitor_state = self.observation_space['monitor'].sample() # use obs_space sampling because its seed is already set

        monitor_obs = self.monitor_state

        monitor_terminated = False
        monitor_truncated = False

        obs = {'mdp': mdp_obs, 'monitor': monitor_obs}
        reward = {'mdp': proxy_reward, 'monitor': monitor_reward}
        terminated = {'mdp': mdp_terminated, 'monitor': monitor_terminated}
        truncated = {'mdp': mdp_truncated, 'monitor': monitor_truncated}
        info = mdp_info | {'mdp_reward': mdp_reward}

        return obs, reward, terminated, truncated, info
