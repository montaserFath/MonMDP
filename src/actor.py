import numpy as np
from abc import ABC, abstractmethod


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
    def __init__(self, critic, init_eps=1., min_eps=0.1, eps_decay=0.0001):
        self._critic = critic
        self._decay = eps_decay
        self._init_eps = init_eps
        self._min_eps = min_eps
        self._train = True
        self.reset()

    def __call__(self, state):
        state = state[0]
        if np.random.random() < self._eps and self._train:
            return [np.random.randint(0, self._critic.n_actions), np.random.randint(0, self._critic.n_mon_actions)]
        else:
            q = self._critic(state)
            encoded_action = q.argmax()
            if encoded_action % 2 == 0:
                mon_action = 0
            else:
                mon_action = 1
            decoded_action = encoded_action // self._critic.n_mon_actions
            return [decoded_action, mon_action]

    def update(self):
        # TODO: _eps should be an object of its own with its decay type, and we just call self._eps.step()
        self._eps = max(self._eps - self._decay, self._min_eps)

    def reset(self):
        self._eps = self._init_eps

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def report(self):
        return self._eps
