from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    The base class that all og the agents are going to be an instance of this class
    """

    def __init__(self, **configs):
        self._n_states = configs["n_states"]
        self._n_actions = configs["n_actions"]

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self.n_actions

    @abstractmethod
    def policy(self, *args, **kwargs):
        """
        Policy of the agent receives a state and outputs the action to be taken
        Parameters
        ----------
        state

        Returns
        -------

        """
        pass

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        """
        Learning procedure of the agent
        Returns
        -------

        """
        pass


class QLearningAgent(BaseAgent):
    """
    The Q-Learning agent
    """

    def __init__(self, **configs):
        super().__init__(**configs)
        self._configs = configs
        self._init_q_values = self._configs["init_q_values"]
        self._q_table = np.ones((self._n_states, self._n_actions)) * self._init_q_values
        self._lr = self._configs["lr"]
        self._gamma = self._configs["gamma"]

    def policy(self, state):
        return np.argmax(self._q_table[state, :])

    def update_policy(self, state, action, reward, done, next_state):
        td_error = reward + self._gamma * np.max(self._q_table[next_state, :]) * (~done) - self._q_table[state, action]
        self._q_table[state, action] += self._lr * td_error
