# pylint: disable=no-member, too-many-arguments, arguments-differ, super-init-not-called
"""Critic for MDP and Monitored MDP"""
from abc import ABC, abstractmethod
import os
import numpy as np
from src.reward import RTable, RDict
from src.replay_buffer import ReplayBuffer
from src.network import NeuralNetowrk


class Critic(ABC):
    """Generic class for the critic"""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        """Update the critic"""
        return

    @abstractmethod
    def reset(self):
        """reset the critic"""
        return

    @abstractmethod
    def report(self):
        """get the current status of the critic"""
        return


# ------------------------------------------------------------------------------
# Classic MDP
# ------------------------------------------------------------------------------


class QCritic(Critic):
    """Critic for Q-learning (as well as SARSA) in MDP"""

    def __init__(self, q0: float = 0.0, gamma=0.99, lr: float = 0.01, on_policy: bool = False, **kwargs):
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._on_policy = on_policy

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        if self._on_policy:  # TODO: on policy is not working fix it
            q_next = self(next_state, next_action)
        else:
            q_next = self(next_state).max()
        target = reward + self._gamma * (1.0 - terminated) * q_next
        prediction = self(state, action)
        new_value = (1.0 - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        return 0.5 * (target - prediction) ** 2

    @property
    def n_actions(self):
        """get the number of the environment actions"""
        return self._n_actions


class QTable(QCritic):
    """Q-Table for critic in MDP"""

    def __init__(
            self,
            observation_space,
            action_space,
            q0: float = 0.0,
            gamma: float = 0.99,
            lr: float = 0.01,
            on_policy: bool = False,
            env_name: str = None,
            **kwargs,
    ):
        QCritic.__init__(self, q0, gamma, lr, on_policy)
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self._q_table = None
        if env_name is not None:
            self._dir_name = "models/{}/q_learning/".format(env_name.split("-")[1])
        self.reset()

    def __call__(self, state, action=None):
        state = state.item() if isinstance(state, np.ndarray) else state
        if action is None:
            return self._q_table[state]
        return self._q_table[state][action]

    def _update(self, state, action, new_value):
        state = state.item() if isinstance(state, np.ndarray) else state
        self._q_table[state][action] = new_value

    def reset(self):
        """reset the values of the q_table to their initial values"""
        shp = (self._n_states, self._n_actions)
        self._q_table = np.ones(shp) * self._q0

    def report(self):
        """get the current q_table values"""
        return self._q_table

    def save(self, seed: int = 1):
        """save q-table as a numpy array"""
        os.makedirs(self._dir_name, exist_ok=True)
        np.save(self._dir_name + "/critic_q_table_{}.npy".format(seed), self._q_table)

    def load(self, log_dir: str = None, seed: int = 1):
        """load the the q-table which saved as a numpy array"""
        if log_dir is None:
            raise ValueError("No files to load Q-Table from it")
        self._q_table = np.load(log_dir + "/critic_q_table_{}.npy".format(seed))


class QDict(QCritic):
    """Dictionary Q for the critic in MDP"""

    def __init__(self, observation_space, action_space, q0=0.0, gamma=0.99, lr=0.01, on_policy=False, **kwargs):
        QCritic.__init__(self, q0, gamma, lr, on_policy)
        self._n_actions = action_space.n
        self._q_dict = None
        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return np.array([q.get(tuple(state), self._q0) for q in self._q_dict])
        else:
            return self._q_dict[action].get(tuple(state), self._q0)

    def _update(self, state, action, new_value):
        self._q_dict[action][tuple(state)] = new_value

    def reset(self):
        """reset the values of the q-dictionary to their initial values"""
        self._q_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        """get the current values of the q-dictionary"""
        return self._q_dict


# ------------------------------------------------------------------------------
# Monitored MDP
# ------------------------------------------------------------------------------

# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches
class MonQCritic(Critic):
    """Dictionary Q for the critic in Monitored MDP"""

    def __init__(
            self,
            env_name: str,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy: str = "reward_model",
            unseen_r_value: float = 0.0,
            **kwargs,
    ):
        self._env_name = env_name
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._on_policy = on_policy
        self._strategy = strategy
        self._unseen_r_value = unseen_r_value
        self._mdp_q = None
        self._mon_q = None
        self._r_model = None
        self._mdp_critic = None


    def update(self, state, action, reward, terminated, next_state, next_action=None):
        """Update the q-value"""
        if not np.isnan(reward["mdp"]):
            if self._strategy == "reward_model":
                self._r_model.update(state["mdp"], action["mdp"], reward["mdp"])
        else:
            if self._strategy == "reward_model":
                reward["mdp"] = self._r_model(state["mdp"], action["mdp"])
            elif self._strategy == "ignore":
                return np.nan, np.nan
            elif self._strategy == "zero_reward":
                reward["mdp"] = self._unseen_r_value
            elif self._strategy in ["q_mdp", "q_monitor_sequential", "q_monitor_joint"]:
                return np.nan, np.nan
            else:
                raise ValueError("unknown update strategy")

        if not np.isnan(reward["mdp"]):
            mdp_error = self._mdp_critic.update(
                state["mdp"], action["mdp"], reward["mdp"], terminated, next_state["mdp"]
            )
        else:
            mdp_error = np.nan

        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            # update Q for MDP and MonMDP
            q_next = self(next_state, next_action if self._on_policy else None)
            prediction = self(state, action)
            new_value, error = {}, {}
            for key in q_next.keys():
                q_next_value = q_next[key].max() if isinstance(q_next[key], np.ndarray) else q_next[key].item()
                target = reward[key] + self._gamma * (1.0 - terminated) * q_next_value
                new_value[key] = (1.0 - self._lr) * prediction[key] + self._lr * target
                error[key] = 0.5 * (target - prediction[key]) ** 2
            self._update(state, action, new_value)
            return error["mdp"], error["monitor"]

        if not np.isnan(reward["mdp"]):
            joint_reward = reward["monitor"] + reward["mdp"]
        else:
            joint_reward = reward["monitor"] + 0.0

        if self._on_policy:
            q_next = self(next_state, next_action).item()
        else:
            q_next = self(next_state).max()
        target = joint_reward + self._gamma * (1.0 - terminated) * q_next
        prediction = self(state, action)
        new_value = (1.0 - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        mon_error = 0.5 * (target - prediction) ** 2

        return mdp_error, mon_error

    @property
    def n_actions(self):
        """get the number of environment actions"""
        return self._n_actions

    @property
    def n_mon_actions(self):
        """get th number of the monitored actions"""
        return self._n_mon_actions


class MonQTable(MonQCritic):
    """Q-Table for critic in Monitored MDP"""

    def __init__(
            self,
            env_name,
            observation_space,
            action_space,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy="zero_reward",
            unseen_r_value=0.0,
            **kwargs,
    ):
        MonQCritic.__init__(self, env_name, q0, gamma, lr, on_policy, strategy=strategy, unseen_r_value=unseen_r_value)
        self._mdp_critic = QTable(observation_space["mdp"], action_space["mdp"], q0, gamma, lr)
        self._n_states = observation_space["mdp"].n
        self._n_actions = action_space["mdp"].n
        self._n_mon_states = observation_space["monitor"].n
        self._n_mon_actions = action_space["monitor"].n
        self._q_table = None
        env_name = self._env_name.split("/")[1].split("-")[1]
        self._dir_name = "models/{}/{}/".format(env_name, self._strategy)

        if self._strategy == "reward_model":
            self._r_model = RTable(observation_space["mdp"], action_space["mdp"], **kwargs["reward_model"])
        else:
            self._r_model = None

        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state["mdp"]][state["monitor"]]
        return self._q_table[state["mdp"]][state["monitor"]][action["mdp"]][action["monitor"]]

    def _update(self, state, action, new_value):
        self._q_table[state["mdp"]][state["monitor"]][action["mdp"]][action["monitor"]] = new_value

    def reset(self):
        """reset the values of the q-table to initial values in Monitored MDP"""
        shp = (self._n_states, self._n_mon_states, self._n_actions, self._n_mon_actions)
        self._q_table = np.ones(shp) * self._q0
        self._mdp_critic.reset()

    def report(self):
        """get the values of the q-table in monitored MDP"""
        return self._q_table, self._mdp_critic.report()


class MonQTableOneAction(MonQTable):
    """Q-table for Monitored MDP with a single monitoring action"""

    def __int__(self, env_name, observation_space, action_space, **kwargs):
        super().__int__(self, env_name, observation_space, action_space)

    def reset(self):
        """reset the values of the q-table to initial values in Monitored MDP"""
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q = np.ones((self._n_states, self._n_actions)) * self._q0
            self._mon_q = np.ones((self._n_states, self._n_actions * self._n_mon_actions)) * 0  # TODO: remove 0
        else:
            self._q_table = np.ones((self._n_states, self._n_actions * self._n_mon_actions)) * self._q0
        self._mdp_critic.reset()

    def __call__(self, state, action=None):
        mdp_s = state["mdp"]
        if action is None:
            if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
                mdp_q = self.expand_mdp_q()[mdp_s] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_s]
                return {"mdp": mdp_q, "monitor": self._mon_q[mdp_s]}
            return self._q_table[mdp_s]

        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q = self.expand_mdp_q()[mdp_s] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_s]
            mon_q = self._mon_q[mdp_s, self.get_action_ind(action)]
            return {"mdp": np.squeeze(mdp_q)[action["mdp"]], "monitor": mon_q}
        return self._q_table[mdp_s, self.get_action_ind(action)]

    def _update(self, state, action, new_value):
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q[state["mdp"], action["mdp"]] = new_value["mdp"]
            self._mon_q[state["mdp"], self.get_action_ind(action)] = new_value["monitor"]
        else:
            self._q_table[state["mdp"], self.get_action_ind(action)] = new_value

    def get_action_ind(self, action: dict) -> int:
        """convert a dictionary of MDP and Monitor actions to an index"""
        mdp_action, mon_action = action["mdp"], action["monitor"]
        return mon_action * self._n_actions + mdp_action

    def ind_to_action(self, action_ind: int) -> dict:
        """convert the action index to a dictionary of MDP and Monitor actions"""
        mon_action, mdp_action = action_ind // self._n_actions, action_ind % self._n_actions
        return {"mdp": mdp_action, "monitor": mon_action}

    def save(self, seed: int = 1):
        """save the q-table as numpy array"""
        os.makedirs(self._dir_name, exist_ok=True)
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            np.save(self._dir_name + "/mdp_q_table_{}.npy".format(seed), self._mdp_q)
            np.save(self._dir_name + "/monitor_q_table_{}.npy".format(seed), self._mon_q)
        else:
            np.save(self._dir_name + "/critic_q_table_{}.npy".format(seed), self._q_table)
        if self._r_model is not None:
            self._r_model.save(self._dir_name)

    def load(self, log_dir: str = None, seed: int = 1):
        """Load a q-table which saved as a numpy array"""
        if log_dir is None:
            raise ValueError("No files to load Q-Table from it")
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q = np.load(self._dir_name + "/mdp_q_table_{}.npy".format(seed))
            self._mon_q = np.load(self._dir_name + "/monitor_q_table_{}.npy".format(seed))
        else:
            self._q_table = np.load(log_dir + "/critic_q_table_{}.npy".format(seed))
        if self._r_model is not None:
            self._r_model = np.load(log_dir + "/reward_model_table_{}.npy".format(seed))

    def expand_mdp_q(self):
        """Expand Q-table for MDP"""
        new_q = np.zeros((self._mdp_q.shape[0], self._mdp_q.shape[1] * 2))
        new_q[:, : self._n_actions] = self._mdp_q.copy()
        new_q[:, self._n_actions :] = self._mdp_q.copy()
        return new_q


class StateMonTable(MonQTableOneAction):
    """Q-table for Monitored MDP with a monitoring state"""

    def reset(self):
        """reset the q-table in a state monitor MDP to initial value"""
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q = np.ones((self._n_states, self._n_actions)) * self._q0
            self._mon_q = np.ones((self._n_states * self._n_mon_states, self._n_actions)) * 0  # TODO: remove 0
        else:
            self._q_table = np.ones((self._n_states * self._n_mon_states, self._n_actions)) * self._q0
        self._mdp_critic.reset()

    def __call__(self, state, action=None):
        mdp_state, mdp_action = state["mdp"].item(), action["mdp"] if action is not None else None
        state_ind = self.get_state_ind(state)

        if action is None:
            if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
                mdp_q = (
                    self.expand_mdp_q()[state_ind] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_state]
                )
                return {"mdp": mdp_q, "monitor": self._mon_q[state_ind]}
            return self._q_table[state_ind]

        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            mdp_q = self.expand_mdp_q()[state_ind] if self._strategy == "q_monitor_joint" else self._mdp_q[mdp_state]
            mon_q = self._mon_q[state_ind, mdp_action]
            return {"mdp": np.squeeze(mdp_q)[mdp_action], "monitor": mon_q}
        return self._q_table[self.get_state_ind(state), mdp_action]

    def _update(self, state, action, new_value):
        if self._strategy in ["q_monitor_sequential", "q_monitor_joint"]:
            self._mdp_q[state["mdp"], action["mdp"]] = new_value["mdp"]
            self._mon_q[self.get_state_ind(state), action["mdp"]] = new_value["monitor"]
        else:
            self._q_table[self.get_state_ind(state), action["mdp"]] = new_value

    def get_state_ind(self, state: dict) -> int:
        """convert MDP and Monitor states to an index"""
        return state["monitor"] * self._n_states + state["mdp"].item()

    def ind_to_state(self, state_ind: int) -> dict:
        """convert state index to a dictionary of MDP and Monitor states"""
        if state_ind >= self._n_states * self._n_mon_states:
            raise ValueError("State index is larger than max Number of states")
        mon_state, mdp_state = state_ind // self._n_states, state_ind % self._n_states
        return {"mdp": mdp_state, "monitor": mon_state}

    def expand_mdp_q(self):
        """Expand Q-table for MDP"""
        new_q = np.zeros((self._mdp_q.shape[0] * 2, self._mdp_q.shape[1]))
        new_q[: self._n_states] = self._mdp_q.copy()
        new_q[self._n_states :] = self._mdp_q.copy()
        return new_q


# pylint: disable=too-many-instance-attributes
class MonQDict(MonQCritic):
    """Q-Dictionary for Monitored MDP"""

    def __init__(
            self,
            observation_space,
            action_space,
            q0=0.0,
            gamma=0.99,
            lr=0.01,
            on_policy=False,
            strategy="zero_reward",
            **kwargs,
    ):
        MonQCritic.__init__(self, q0, gamma, lr, on_policy)
        self._mdp_critic = QDict(observation_space["mdp"], action_space["mdp"], q0, gamma, lr)
        self._n_actions = action_space["mdp"].n
        self._n_mon_actions = action_space["monitor"].n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self._q_dict = None

        if self._strategy == "reward_model":
            self._r_model = RDict(observation_space["mdp"], action_space["mdp"], **kwargs["reward_model"])
        else:
            self._r_model = None

        self.reset()

    def __call__(self, state, action=None):
        state_full = np.concatenate((state["mdp"], [state["monitor"]]))

        if action is None:
            return np.array([[q_mon.get(tuple(state_full), self._q0) for q_mon in q_mdp] for q_mdp in self._q_dict])
        return self._q_dict[action["mdp"]][action["monitor"]].get(tuple(state), self._q0)

    def _update(self, state, action, new_value):
        state = np.concatenate((state["mdp"], [state["monitor"]]))
        self._q_dict[action["mdp"]][action["monitor"]][tuple(state)] = new_value

    def reset(self):
        """reseat the Q-dictionary to initial value"""
        self._q_dict = [[dict() for _ in range(self._n_mon_actions)] for _ in range(self._n_actions)]
        self._mdp_critic.reset()

    def report(self):
        """get the current values of Q-dictionary"""
        return self._q_dict, self._mdp_critic.report()


class MonQNet(MonQCritic):
    def __int__(
            self, env_name, q0, gamma, lr, on_policy, strategy, unseen_r_value, observation_space, action_space,
    ):
        MonQCritic.__init__(self, env_name, q0, gamma, lr, on_policy, strategy=strategy, unseen_r_value=unseen_r_value)
        self.replay_buffer = ReplayBuffer(observation_space.shape, action_space.shape)
        self.network = NeuralNetowrk(observation_space.shape, action_space.n)

    def reset(self):
        NotImplemented

    def train(self):
        NotImplemented

    def predict(self):
        NotImplemented

    def __call__(self, state, action=None):
        NotImplemented

    def save(self):
        NotImplemented

    def load(self):
        NotImplemented

    def update(self, state, action, reward, terminated, next_state, next_action=None):
        NotImplemented



