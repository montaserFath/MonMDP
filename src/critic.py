import numpy as np
from abc import ABC, abstractmethod
from src.reward import RTable, RDict


class Critic(ABC):
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



# ------------------------------------------------------------------------------
# Classic MDP
# ------------------------------------------------------------------------------

class QCritic(Critic):
    def update(self, state, action, reward, terminated, next_state):
        q_next = self(next_state)
        target = reward + self._gamma * (1. - terminated) * q_next.max()
        prediction = self(state, action)
        new_value = (1. - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        return 0.5 * (target - prediction) ** 2

    @property
    def n_actions(self):
        return self._n_actions


class QTable(QCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, **kwargs):
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state]
        else:
            return self._q_table[state][action]

    def _update(self, state, action, new_value):
        self._q_table[state][action] = new_value

    def reset(self):
        self._q_table = np.ones((self._n_states, self._n_actions)) * self._q0

    def report(self):
        return self._q_table


class QDict(QCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, **kwargs):
        self._n_actions = action_space.n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return np.array([
                q.get(tuple(state), self._q0) for q in self._q_dict
            ])
        else:
            return self._q_dict[action].get(tuple(state), self._q0)

    def _update(self, state, action, new_value):
        self._q_dict[action][tuple(state)] = new_value

    def reset(self):
        self._q_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        return self._q_dict




# ------------------------------------------------------------------------------
# Monitored MDP
# ------------------------------------------------------------------------------

class MonQCritic(Critic):
    def update(self, state, action, reward, terminated, next_state):
        if not np.isnan(reward['mdp']):
            if self._r_model is not None:
                self._r_model.update(state['mdp'], action['mdp'], reward['mdp'])
                reward['mdp'] = self._r_model(state['mdp'], action['mdp'])

        if not np.isnan(reward['mdp']):
            mdp_error = self._mdp_critic.update(
                state['mdp'], action['mdp'], reward['mdp'], terminated, next_state['mdp'])
        else:
            mdp_error = np.nan

        if not np.isnan(reward['mdp']):
            reward = reward['monitor'] + reward['mdp']
        else:
            reward = reward['monitor']

        q_next = self(next_state)
        target = reward + self._gamma * (1. - terminated) * q_next.max()
        prediction = self(state, action)
        new_value = (1. - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        mon_error = 0.5 * (target - prediction) ** 2

        return mdp_error, mon_error

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def n_mon_actions(self):
        return self._n_mon_actions


class MonQTable(MonQCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, use_reward_model=True, **kwargs):
        self._mdp_critic = QTable(observation_space['mdp'], action_space['mdp'], q0, gamma, lr)
        self._n_states = observation_space['mdp'].n
        self._n_actions = action_space['mdp'].n
        self._n_mon_states = observation_space['monitor'].n
        self._n_mon_actions = action_space['monitor'].n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        if use_reward_model:
            self._r_model = RTable(observation_space['mdp'], action_space['mdp'], q0, lr)
        else:
            self._r_model = None
        self.reset()

    def __call__(self, state, action=None):
        if action is None:
            return self._q_table[state['mdp']][state['monitor']]
        else:
            return self._q_table[state['mdp']][state['monitor']][action['mdp']][action['monitor']]

    def _update(self, state, action, new_value):
        self._q_table[state['mdp']][state['monitor']][action['mdp']][action['monitor']] = new_value

    def reset(self):
        self._q_table = np.ones((self._n_states, self._n_mon_states, self._n_actions, self._n_mon_actions)) * self._q0
        self._mdp_critic.reset()

    def report(self):
        return self._q_table, self._mdp_critic.report()


class MonQDict(MonQCritic):
    def __init__(self, observation_space, action_space,
                 q0=0., gamma=0.99, lr=0.01, use_reward_model=True, **kwargs):
        self._mdp_critic = QDict(observation_space['mdp'], action_space['mdp'], q0, gamma, lr)
        self._n_actions = action_space['mdp'].n
        self._n_mon_actions = action_space['monitor'].n
        self._q0 = q0
        self._gamma = gamma
        self._lr = lr
        if use_reward_model:
            self._r_model = RDict(observation_space['mdp'], action_space['mdp'], q0, lr)
        else:
            self._r_model = None
        self.reset()

    def __call__(self, state, action=None):
        state_full = np.concatenate((state['mdp'], [state['monitor']]))

        if action is None:
            return np.array([
                [q_mon.get(tuple(state_full), self._q0) for q_mon in q_mdp]
                for q_mdp in self._q_dict
            ])
        else:
            return self._q_dict[action['mdp']][action['monitor']].get(
                tuple(state),
                self._q0
            )

    def _update(self, state, action, new_value):
        state = np.concatenate((state['mdp'], [state['monitor']]))
        self._q_dict[action['mdp']][action['monitor']][tuple(state)] = new_value

    def reset(self):
        self._q_dict = [
            [dict() for _ in range(self._n_mon_actions)]
            for _ in range(self._n_actions)
        ]
        self._mdp_critic.reset()

    def report(self):
        return self._q_dict, self._mdp_critic.report()
