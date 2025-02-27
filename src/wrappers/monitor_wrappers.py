"""Monitor wrappers for different scenarios"""
from abc import abstractmethod
import gymnasium
from gymnasium import spaces
import numpy as np


class Monitor(gymnasium.Wrapper):
    """
    Generic monitor class.

    Args:
        env (gymnasium.Env): the Gymnasium environment.

    """

    @abstractmethod
    def _monitor_step(self, action, mdp_reward, mdp_state=None, mdp_info=None):
        pass

    def step(self, action):
        mdp_obs, mdp_reward, mdp_terminated, mdp_truncated, mdp_info = self.env.step(action["mdp"])
        monitor_obs, proxy_reward, monitor_cost = self._monitor_step(action, mdp_reward, mdp_obs, mdp_info)

        obs = {"mdp": mdp_obs, "monitor": monitor_obs}
        reward = {"mdp": proxy_reward, "monitor": monitor_cost}
        terminated = mdp_terminated
        truncated = mdp_truncated
        info = mdp_info | {"mdp_reward": mdp_reward}

        return obs, reward, terminated, truncated, info


class FullMonitor(Monitor):
    """
    Monitor always shows the true reward without any cost, regardless of states
    and actions.
    This is equivalent to a classic MDP.

    Args:
        env (gymnasium.Env): the Gymnasium environment.

    """

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict(
            {
                "mdp": env.action_space,
                "monitor": spaces.Discrete(1),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "mdp": env.observation_space,
                "monitor": spaces.Discrete(1),
            }
        )

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        monitor_obs = 0  # default monitor state, always active
        return {"mdp": mdp_obs, "monitor": monitor_obs}, mdp_info

    def _monitor_step(self, action, mdp_reward, mdp_state=None, mdp_info=None):
        monitor_cost = 0.0
        proxy_reward = mdp_reward
        monitor_obs = 0
        return monitor_obs, proxy_reward, monitor_cost


class StateMonitor(Monitor):
    """
    State Monitor where the monitor availability depends on the state, and a sequence of env actions can change the
    monitor state
    """

    def __init__(
        self,
        env,
        full_monitor: bool,
        monitor_cost: float = 0.01,
        init_state_prob: float = 0.5,
        button_state: int = 6,
        button_action: int = 1,
        **kwargs,
    ):
        """initialization function"""
        gymnasium.Wrapper.__init__(self, env)
        self.full_monitor = full_monitor
        self.action_space = spaces.Dict({"mdp": env.action_space, "monitor": spaces.Discrete(1)})
        self.observation_space = spaces.Dict({"mdp": env.observation_space, "monitor": spaces.Discrete(2)})
        self.monitor_cost = monitor_cost
        self.init_state_prob = init_state_prob
        self.button_state = button_state
        self.button_action = button_action
        self.monitor_state = None
        self.prev_mdp_state = None

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = (np.random.random() < self.init_state_prob) + 0
        self.prev_mdp_state = None
        return {"mdp": mdp_obs, "monitor": self.monitor_state}, mdp_info

    def _monitor_step(self, action, mdp_reward, mdp_state=None, mdp_info=None):
        if self.prev_mdp_state == self.button_state and action["mdp"] == self.button_action:
            self.monitor_state = (self.monitor_state + 1) % 2  # flip the button

        if self.full_monitor:
            return self.monitor_state, mdp_reward, 0.0
            # return self.monitor_state, mdp_reward, - self.monitor_cost
        if self.monitor_state == 0:  # button is off -> monitor is off
            monitor_cost = 0.0
            proxy_reward = np.nan
        else:  # button is on -> monitor is on
            monitor_cost = -self.monitor_cost
            proxy_reward = mdp_reward
        self.prev_mdp_state = mdp_state
        return self.monitor_state, proxy_reward, monitor_cost


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
        monitor_cost (float): cost for monitor request,
        monitor_reset_prob (float): probability of the monitor resetting itself.

    """

    def __init__(
        self,
        env,
        full_monitor: bool,
        monitor_cost=0.01,
        monitor_reset_prob=0.5,
        init_monitor_state: int = 0,
        empty_observation_space: bool = True,
        **kwargs,
    ):
        """initialization function"""
        gymnasium.Wrapper.__init__(self, env)
        self.full_monitor = full_monitor
        self.action_space = spaces.Dict(
            {
                "mdp": env.action_space,
                "monitor": spaces.Discrete(2),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "mdp": env.observation_space,
                "monitor": spaces.Discrete(2 if not empty_observation_space else 1),
            }
        )
        self.init_monitor_state = init_monitor_state
        self.monitor_state = self.init_monitor_state  # deactivated
        self.monitor_reset_prob = monitor_reset_prob
        self.monitor_cost = monitor_cost

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = self.init_monitor_state
        return {"mdp": mdp_obs, "monitor": self.monitor_state}, mdp_info

    def _monitor_step(self, action, mdp_reward, mdp_state=None, mdp_info=None):
        if self.full_monitor:
            # return 1, mdp_reward, 0.
            return 1, mdp_reward, -self.monitor_cost if action["monitor"] == 1 else 0.0
        if action["monitor"] == 1:
            self.monitor_state = 1
            monitor_cost = -self.monitor_cost
            proxy_reward = mdp_reward
        elif action["monitor"] == 0:
            self.monitor_state = 0
            monitor_cost = 0.0
            proxy_reward = np.nan
        else:
            raise ValueError("illegal monitor action")
        return self.monitor_state, proxy_reward, monitor_cost


class RoomMonitor(Monitor):
    """Divide grid to monitored and unmonitored rooms."""

    def __init__(
        self,
        env,
        full_monitor: bool,
        monitor_cost: float = 0.0,
        monitor_column_ind: int = 3,
        **kwargs,
    ):
        """initialization function"""
        gymnasium.Wrapper.__init__(self, env)
        self.env = env
        self.full_monitor = full_monitor
        self.action_space = spaces.Dict({"mdp": env.action_space, "monitor": spaces.Discrete(1)})
        self.observation_space = spaces.Dict({"mdp": env.observation_space, "monitor": spaces.Discrete(2)})
        self.monitor_cost = monitor_cost
        self.monitor_column_ind = monitor_column_ind
        self.monitor_state = None

    def reset(self, seed=None, **kwargs):
        """reset the environment"""
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        mdp_obs, mdp_info = self.env.reset(seed=seed, **kwargs)
        self.get_monitor_state(mdp_info["previous_agent_pos"])
        return {"mdp": mdp_obs, "monitor": self.monitor_state}, mdp_info

    def _monitor_step(self, action, mdp_reward, mdp_state=None, mdp_info=None):
        if mdp_state is None:
            raise ValueError("mdp_state is None")
        self.get_monitor_state(mdp_info["previous_agent_pos"])
        if mdp_info["previous_agent_pos"][1] < self.monitor_column_ind:
            return self.monitor_state, mdp_reward, 0.0
        return self.monitor_state, np.nan, 0.0

    def get_monitor_state(self, agent_pos):
        self.monitor_state = 1 if agent_pos[1] < self.monitor_column_ind else 0
