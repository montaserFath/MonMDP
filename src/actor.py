# pylint: disable=no-member
"""Different actors including Epsilon greedy for MDP and Monitored-MDP"""
from abc import ABC, abstractmethod
import random
import numpy as np


class LinearEpsilonDecay:
    """Decay the exploration rate linearly every timesteps"""

    def __init__(self, init_eps: float = 1.0, min_eps: float = 0.1, eps_decay: float = 1e-4):
        self._init_value = init_eps
        self._min_value = min_eps
        self._decay = eps_decay
        self._value = init_eps

    def step(self) -> float:
        """decay Epsilon by the decay rate"""
        self._value = max(self._value - self._decay, self._min_value)

    def reset(self) -> float:
        """set Epsilon to the initial value"""
        self._value = self._init_value

    @property
    def value(self) -> float:
        """report the current Epsilon value"""
        return self._value


class Actor(ABC):
    """Generic actor class"""

    @abstractmethod
    def __init__(self, critic, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self):
        """Update the actor"""
        return

    @abstractmethod
    def reset(self):
        """reseat the actor to initial"""
        return

    @abstractmethod
    def report(self):
        """report the current status of the actor"""
        return


class EpsilonGreedy(Actor):
    """Epsilon greedy policy for MDP"""

    # pylint: disable=too-many-arguments
    def __init__(
            self, critic, init_eps: float = 1.0, min_eps: float = 0.1, eps_decay: float = 0.0001, train: bool = True,
    ):
        self._critic = critic
        self._eps = LinearEpsilonDecay(init_eps, min_eps, eps_decay)
        self._train = train
        self.reset()

    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return np.random.randint(0, self._critic.n_actions)
        q_values = self._critic.predict(state)
        indx = np.argwhere(q_values == np.max(q_values))
        # break ties randomly
        return random.choice(indx)[0]
        # return self._critic(state).argmax()

    def update(self):
        """decay Epsilon"""
        self._eps.step()

    def reset(self):
        """reset Epsilon to the initial value"""
        self._eps.reset()

    def eval(self):
        """Evaluate the actor with the greedy policy"""
        self._train = False

    def train(self):
        """train the actor"""
        self._train = True

    def report(self):
        """report the current Epsilon value"""
        return self._eps.value


class MonEpsilonGreedy(EpsilonGreedy):
    """Epsilon greedy actor for Monitored MDP"""

    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return {
                "mdp": np.random.randint(0, self._critic.n_actions),
                "monitor": np.random.randint(0, self._critic.n_mon_actions),
            }
        q_value = self._critic(state)
        action = np.unravel_index(np.argmax(q_value), q_value.shape)
        return {"mdp": action[0], "monitor": action[1]}


class MonEpsilonGreedyOneAction(MonEpsilonGreedy):
    """Epsilon greedy actor for Monitored MDP with a single monitoring action"""

    def ind_to_action(self, action_ind: int) -> dict:
        """Convert action index to a dictionary of MDP action & Monitor action"""
        if action_ind >= (self._critic.n_actions * self._critic.n_mon_actions):
            raise ValueError("action index is larger than max action")
        mon_action, mdp_action = action_ind // self._critic.n_actions, action_ind % self._critic.n_actions
        return {"mdp": mdp_action, "monitor": mon_action}

    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return {
                "mdp": np.random.randint(0, self._critic.n_actions),
                "monitor": np.random.randint(0, self._critic.n_mon_actions),
            }
        q_value = self._critic(state)
        if isinstance(q_value, dict):
            q_mdp = np.squeeze(q_value["mdp"])
            q_mon = np.squeeze(q_value["monitor"])
            if self._critic._strategy == "q_monitor_sequential":
                mdp_action = np.argmax(q_mdp)
                mon_action = [q_mon[mdp_action], q_mon[mdp_action + self._critic.n_actions]]
                return {"mdp": mdp_action, "monitor": np.argmax(mon_action)}
            return self.ind_to_action(np.argmax(q_mdp + q_mon))
        return self.ind_to_action(np.argmax(q_value))


class MonStateEpsilonGreedy(MonEpsilonGreedyOneAction):
    """Epsilon greedy actor for Monitored MDP with state monitoring"""

    # pylint: disable=protected-access
    def ind_to_state(self, state_ind: int) -> dict:
        """convert state index to dictionary of MDP state and Monitor state"""
        if state_ind >= self._critic.n_states * self._critic.n_mon_states:
            raise ValueError("State index is larger than max Number of states")
        mon_state, mdp_state = state_ind // self._critic._n_states, state_ind % self._critic._n_states
        return {"mdp": mdp_state, "monitor": mon_state}

    def get_state_ind(self, state: dict) -> int:
        """convert a dictionary of MDP state and Monitor state to an index"""
        return state["monitor"] * self._critic._n_states + state["mdp"].item()

    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return {"mdp": np.random.randint(0, self._critic.n_actions), "monitor": 0}
        q_value = self._critic(state)
        if isinstance(q_value, dict):
            q_mdp = np.squeeze(q_value["mdp"])
            q_mon = np.squeeze(q_value["monitor"])
            if self._critic._strategy == "q_monitor_sequential":
                return {"mdp": np.argmax(q_mdp), "monitor": 0}
            return {"mdp": np.argmax(q_mdp + q_mon), "monitor": 0}
        return {"mdp": np.argmax(q_value), "monitor": 0}
