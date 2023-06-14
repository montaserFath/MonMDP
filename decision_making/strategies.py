import numpy as np
from .agents import BaseAgent
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_action(self, *args, **kwargs):
        pass


class EpsilonGreedy:
    def __init__(self, agent: BaseAgent, eps=1.0, decay=0.01, min_eps=0):
        self._agent = agent
        self._eps = eps
        self._decay = decay
        self._min_eps = min_eps

    def select_action(self, state):
        if np.random.random() < self._eps:
            return np.random.randint(0, self._agent.n_actions)
        else:
            return self._agent.policy(state)

    def linear_decay(self):
        self._eps = self._eps - self._decay if self._eps - self._decay > self._min_eps else self._min_eps
