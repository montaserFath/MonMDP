import numpy as np
from abc import ABC, abstractmethod


class LinearEpsilonDecay:
    def __init__(self, init_eps=1., min_eps=0.1, eps_decay=1e-4):
        self._init_value = init_eps
        self._min_value = min_eps
        self._decay = eps_decay
        self._value = init_eps

    def step(self):
        self._value = max(self._value - self._decay, self._min_value)

    def reset(self):
        self._value = self._init_value

    @property
    def value(self):
        return self._value


class Actor(ABC):
    @abstractmethod
    def __init__(self, critic, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
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


class EpsilonGreedy(Actor):
    def __init__(self, critic, init_eps=1., min_eps=0.1, eps_decay=0.0001, train: bool = True):
        self._critic = critic
        self._eps = LinearEpsilonDecay(init_eps, min_eps, eps_decay)
        self._train = train
        self.reset()

    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return np.random.randint(0, self._critic.n_actions)
        else:
            return self._critic(state).argmax()

    def update(self):
        self._eps.step()

    def reset(self):
        self._eps.reset()

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def report(self):
        return self._eps.value


class MonEpsilonGreedy(EpsilonGreedy):
    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return {'mdp': np.random.randint(0, self._critic.n_actions),
                    'monitor': np.random.randint(0, self._critic.n_mon_actions)}
        else:
            q = self._critic(state)
            action = np.unravel_index(np.argmax(q), q.shape)
            return {'mdp': action[0], 'monitor': action[1]}


class MonEpsilonGreedyOneAction(MonEpsilonGreedy):
    def ind_to_action(self, action_ind: int) -> dict:
        if action_ind >= (self._critic.n_actions * self._critic.n_mon_actions):
            raise ValueError("action index is larger than max action")
        mon_action, mdp_action = action_ind // self._critic.n_actions, action_ind % self._critic.n_actions
        return {"mdp": mdp_action, "monitor": mon_action}

    def __call__(self, state):
        if np.random.random() < self._eps.value and self._train:
            return {'mdp': np.random.randint(0, self._critic.n_actions),
                    'monitor': np.random.randint(0, self._critic.n_mon_actions)}
        q = self._critic(state)
        if self._critic._strategy == "q_mdp":
            return {"mdp": np.argmax(q), "monitor": 1}
        return self.ind_to_action(np.argmax(q))
