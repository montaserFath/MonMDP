from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    The base class that all og the agents are going to be an instance of this class
    """

    def __init__(self, **params):
        self._n_states = params["n_states"]
        self._n_actions = params["n_actions"]

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self._n_actions

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

    @abstractmethod
    def reset(self):
        pass


class QLearningAgent(BaseAgent):
    """
    The Q-Learning agent
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._params = params
        self._init_q_values = self._params["init_q_values"]
        self._q_table = np.ones((self._n_states, self._n_actions)) * self._init_q_values
        self._lr = self._params["lr"]
        self._gamma = self._params["gamma"]

    def policy(self, state):
        return np.argmax(self._q_table[state, :])

    def update_policy(self, state, action, reward, done, next_state):
        td_error = reward + self._gamma * np.max(self._q_table[next_state, :]) * (1 - done) - self._q_table[
            state, action]
        self._q_table[state, action] += self._lr * td_error

    def reset(self):
        self._q_table = np.ones((self._n_states, self._n_actions)) * self._init_q_values
