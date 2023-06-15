import numpy as np
from .agents import BaseAgent
from abc import ABC, abstractmethod
from typing import Callable


class BaseStrategy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def report(self):
        pass


class EpsilonGreedy(BaseStrategy):
    def __init__(self, agent: BaseAgent, **params):
        self._params = params
        self._agent = agent
        self._eps = self._params["eps"]
        self._decay = self._params["decay"]
        self._min_eps = self._params["min_eps"]

    def select_action(self, state):
        if np.random.random() < self._eps:
            return np.random.randint(0, self._agent.n_actions)
        else:
            return self._agent.policy(state)

    def linear_decay(self):
        self._eps = self._eps - self._decay if self._eps - self._decay > self._min_eps else self._min_eps

    def update(self):
        if self._params["decay_type"] == "linear":
            self.linear_decay()

    def reset(self):
        self._eps = self._params["eps"]

    def report(self):
        return self._eps
