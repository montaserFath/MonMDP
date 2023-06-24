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
        self.reset()

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
        self.reset()

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
        self._n_actions = action_space['mdp'].n
        self._n_mon_actions = action_space['monitor'].n
        self._init_q_values = init_q_values
        self._gamma = gamma
        self._lr = lr

    def __call__(self, state, action=None):
        mdp_state = state['mdp']
        mon_state = state['monitor']
        state = np.concatenate((mdp_state, [mon_state]))

        if action is None:
            return np.array([
                [q_mon.get(tuple(state), self._init_q_values) for q_mon in q_mdp] for q_mdp in self._mon_q_dict
            ])
        else:
            mdp_action = action['mdp']
            mon_action = action['monitor']
            return self._q_dict[mdp_action][mon_action].get(tuple(state), self._init_q_values)

    def update(self, state, action, reward, terminated, next_state):
        mdp_state = state['mdp']
        mon_state = state['monitor']
        mdp_action = action['mdp']
        mon_action = action['monitor']
        mdp_reward = reward['mdp']
        mon_reward = reward['monitor']
        mdp_terminated = terminated['mdp']
        mon_terminated = terminated['monitor']
        mdp_next_state = next_state['mdp']
        mon_next_state = next_state['monitor']

        if mdp_reward is not np.nan:
            mdp_q_next = np.array([
                q.get(tuple(tuple(mdp_next_state)), self._init_q_values) for q in self._q_dict
            ])
            mdp_target = mdp_reward + self._gamma * (1. - mdp_terminated) * mdp_q_next.max()
            mdp_prediction = self._q_dict[mdp_action].get(tuple(mdp_state), self._init_q_values)
            self._q_dict[mdp_action][tuple(mdp_state)] = (1. - self._lr) * mdp_prediction + self._lr * mdp_target
            mdp_error = 0.5 * (mdp_target - mdp_prediction) ** 2
        else:
            mdp_error = np.nan

        next_state = np.concatenate((mdp_next_state, [mon_next_state]))
        state = np.concatenate((mdp_state, [mon_state]))
        reward = mon_reward
        if mdp_reward is not np.nan:
            reward += mdp_reward

        mon_q_next = np.array([
            [q_mon.get(tuple(next_state), self._init_q_values) for q_mon in q_mdp] for q_mdp in self._mon_q_dict
        ])
        mon_target = reward + self._gamma * (1. - mdp_terminated) * mon_q_next.max()
        mon_prediction = self._mon_q_dict[mdp_action][mon_action].get(tuple(state), self._init_q_values)
        self._mon_q_dict[mdp_action][mon_action][tuple(state)] = (1. - self._lr) * mon_prediction + self._lr * mon_target
        mon_error = 0.5 * (mon_target - mon_prediction) ** 2

        return mdp_error, mon_error

    def reset(self):
        self._q_dict = [dict() for _ in range(self._n_actions)]
        self._mon_q_dict = [[dict() for _ in range(self._n_mon_actions)] for _ in range(self._n_actions)]

    def report(self):
        return self._q_dict, self._mon_q_dict
