import numpy as np
from abc import ABC, abstractmethod


class Critic(ABC):
    @abstractmethod
    def __init__(self, observation_space, action_space, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def report(self):
        pass

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_mon_actions(self):
        return self._n_mon_actions


class QTable(Critic):
    def __init__(self, observation_space, action_space, init_q_values=0., gamma=0.99, lr=0.01):
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self._init_q_values = init_q_values
        self._gamma = gamma
        self._lr = lr

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state]
        else:
            return self._q_table[state][action]

    def update(self, state, action, reward, terminated, next_state):
        target = reward + self._gamma * (1. - terminated) * self._q_table[next_state].max()
        prediction = self._q_table[state][action]
        self._q_table[state][action] = (1. - self._lr) * prediction + self._lr * target
        return 0.5 * (target - prediction) ** 2

    def reset(self):
        self._q_table = np.ones((self._n_states, self._n_actions)) * self._init_q_values

    def report(self):
        return self._q_table


class QDict(Critic):
    def __init__(self, observation_space, action_space, init_q_values=0., gamma=0.99, lr=0.01):
        self._n_actions = action_space.n
        self._init_q_values = init_q_values
        self._gamma = gamma
        self._lr = lr

    def __call__(self, state, action=None):
        if action is None:
            return np.array([
                q.get(tuple(state), self._init_q_values) for q in self._q_dict
            ])
        else:
            return self._q_dict[action].get(tuple(state), self._init_q_values)

    def update(self, state, action, reward, terminated, next_state):
        q_next = np.array([
            q.get(tuple(next_state), self._init_q_values) for q in self._q_dict
        ])
        target = reward + self._gamma * (1. - terminated) * q_next.max()
        prediction = self._q_dict[action].get(tuple(state), self._init_q_values)
        self._q_dict[action][tuple(state)] = (1. - self._lr) * prediction + self._lr * target
        return 0.5 * (target - prediction) ** 2

    def reset(self):
        self._q_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        return self._q_dict


class MonQDict(Critic):
    def __init__(self, observation_space, action_space, init_q_values=0., gamma=0.99, lr=0.01):
        self._n_actions = action_space[0].n
        self._n_mon_actions = action_space[1].n
        self._init_q_values = init_q_values
        self._gamma = gamma
        self._lr = lr
        self._q_dict = [[dict() for _ in range(self._n_mon_actions)] for _ in range(self._n_actions)]

    def __call__(self, state, action=None):
        if action is None:
            return np.array([
                q.get(tuple(state), self._init_q_values) for inner_q in self._q_dict for q in inner_q
            ])
        else:
            reg_action, mon_action = action
            return self._q_dict[reg_action][mon_action].get(tuple(state), self._init_q_values)

    def update(self, state, action, reward, terminated, next_state):
        q_next = np.array([
            q.get(tuple(next_state), self._init_q_values) for inner_q in self._q_dict for q in inner_q
        ])
        target = reward + self._gamma * (1. - terminated) * q_next.max()
        prediction = self._q_dict[action[0]][action[1]].get(tuple(state), self._init_q_values)
        self._q_dict[action[0]][action[1]][tuple(state)] = (1. - self._lr) * prediction + self._lr * target

    def reset(self):
        self._q_dict = [[dict() for _ in range(self._n_mon_actions)] for _ in range(self._n_actions)]

    def report(self):
        return self._q_dict
