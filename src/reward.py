import numpy as np
from abc import ABC, abstractmethod


class Reward(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def report(self):
        pass

    def update(self, state, action, reward):
        target = reward
        prediction = self(state, action)
        new_value = (1. - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        return 0.5 * (target - prediction) ** 2

    @property
    def n_actions(self):
        return self._n_actions


class RTable(Reward):
    def __init__(self, observation_space, action_space, r0=0., lr=0.01):
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self._r0 = r0
        self._lr = lr
        self.reset()

    def __call__(self, state, action):
        return self._r_table[state][action]

    def _update(self, state, action, new_value):
        self._r_table[state][action] = new_value

    def reset(self):
        self._r_table = np.ones((self._n_states, self._n_actions)) * self._r0

    def report(self):
        return self._r_table


class RDict(Reward):
    def __init__(self, observation_space, action_space, r0=0., lr=0.01):
        self._n_actions = action_space.n
        self._r0 = r0
        self._lr = lr
        self.reset()

    def __call__(self, state, action):
        return self._r_dict[action].get(tuple(state), self._r0)

    def _update(self, state, action, new_value):
        self._r_dict[action][tuple(state)] = new_value

    def reset(self):
        self._r_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        return self._r_dict
