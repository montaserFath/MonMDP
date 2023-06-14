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

        self.mdp_action_space = env.action_space
        self.monitor_action_space = spaces.Discrete(2)
        self.action_space = spaces.MultiDiscrete([
            self.mdp_action_space.n,
            self.monitor_action_space.n,
        ])

        self.mdp_observation_space = env.observation_space
        self.monitor_observation_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            'mdp': self.mdp_observation_space,
            'monitor': self.monitor_observation_space,
        })

        self.monitor_state = 0 # deactivated

    def reset(self, seed=None, **kwargs):
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_observation_space.seed(seed)
        self.monitor_action_space.seed(seed)
        self.monitor_state = 0
        return (mdp_obs, self.monitor_state), mdp_info

    def step(self, action):
        mdp_action, monitor_action = action

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
            self.monitor_state = self.monitor_observation_space.sample() # use obs_space sampling because its seed is already set

        monitor_obs = self.monitor_state

        monitor_terminated = False
        monitor_truncated = False

        obs = (mdp_obs, monitor_obs)
        reward = (proxy_reward, monitor_reward)
        terminated = (mdp_terminated, monitor_terminated)
        truncated = (mdp_truncated, monitor_truncated)
        info = mdp_info | {'mdp_reward': mdp_reward}

        return obs, reward, terminated, truncated, info
