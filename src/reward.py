"""Predictive Reward Model"""
from abc import ABC, abstractmethod
import numpy as np
from src.network import CNN
import torch


class Reward(ABC):
    """Generic reward class"""

    @abstractmethod
    def __init__(self, lr: float = 1.0, **kwargs):
        self._lr = lr

    @abstractmethod
    def __call__(self, **kwargs):
        return

    @abstractmethod
    def reset(self):
        """reset the reward model"""
        return

    @abstractmethod
    def report(self):
        """get the reward model"""
        return

    def update(self, state, action, reward):
        """update the reward value for the state-action pair"""
        target = reward
        prediction = self(state, action)
        new_value = (1.0 - self._lr) * prediction + self._lr * target
        self._update(state, action, new_value)
        return 0.5 * (target - prediction) ** 2

    @property
    def n_actions(self):
        return self._n_actions


class RTable(Reward):
    """Reward table for Environment reward"""

    def __init__(self, observation_space, action_space, r0=0.0, lr=0.01, **kwargs):
        self._n_states = observation_space.n
        self._n_actions = action_space.n
        self._r0 = r0
        self._lr = lr
        self.reset()

    def __call__(self, state, action):
        state = state.item() if isinstance(state, np.ndarray) else state
        return self._r_table[state][action]

    def _update(self, state, action, new_value):
        state = state.item() if isinstance(state, np.ndarray) else state
        old_value = self._r_table[state][action]
        self._r_table[state][action] = old_value + self._lr * (new_value - old_value)

    def reset(self):
        """reset the reward table to the initialization value"""
        self._r_table = np.ones((self._n_states, self._n_actions)) * self._r0

    def report(self):
        """get the reward table"""
        return self._r_table

    def save(self, log_dir: str = None, seed: int = 1):
        """save the reward table as a numpy array"""
        if log_dir is None:
            raise ValueError("The log directory is empty")
        np.save(log_dir + "/reward_model_table_{}.npy".format(seed), self._r_table)


class RDict(Reward):
    """Reward dictionary for Environment reward"""

    def __init__(self, observation_space, action_space, r0=0.0, lr=0.01, **kwargs):
        self._n_actions = action_space.n
        self._r0 = r0
        self._lr = lr
        self.reset()

    def __call__(self, state, action):
        return self._r_dict[action].get(tuple(state), self._r0)

    def _update(self, state, action, new_value):
        self._r_dict[action][tuple(state)] = new_value

    def reset(self):
        """reset the reward dictionary to the initialization value"""
        self._r_dict = [dict() for _ in range(self._n_actions)]

    def report(self):
        """get the reward dictionary"""
        return self._r_dict


class RewardNet(Reward):
    def __init__(self, observation_space, action_space, lr: float = 0.01, device: str = None, **kwargs):
        self._obs_size = observation_space.shape
        self._n_actions = action_space.n
        self._lr = lr
        self._device = device
        self._network = CNN(self._obs_size, self._n_actions, self._lr, device=self._device)
        self._network.init_network()
        self._reward_optimizer = torch.optim.Adam(self._network.model.parameters(), lr=self._lr)
        self._reward_loss_fun = torch.nn.SmoothL1Loss()
        self._reward_loss = []

    def __call__(self, state, action):
        return self.inference(state, action)

    def optimize_reward_model(self, batch: dict):
        real_reward_idx = batch["real_reward_idx"]
        mdp_obs = batch["mdp_obs"][real_reward_idx]
        mdp_action = batch["mdp_action"][real_reward_idx]
        target_reward = batch["mdp_reward"][real_reward_idx]
        expected_reward = self._network(mdp_obs).gather(1, mdp_action)
        loss = self._reward_loss_fun(target_reward, expected_reward)

        self._reward_optimizer.zero_grad()
        loss.backward()
        self._reward_optimizer.step()
        self._reward_loss.append(loss.item())
        return loss.item()

    def inference(self, state: np.ndarray, actions: np.ndarray = None) -> torch.Tensor:
        state_tensor = torch.from_numpy(state).float().to(self._device)
        if actions is None:
            return self._network.forward(state_tensor)
        actions_tensor = torch.from_numpy(actions).float().to(self._device)
        return self._network.forward(state_tensor).gather(1, actions_tensor)

    def reset(self):
        self._network.init_network()
        self._reward_optimizer = torch.optim.Adam(self._network.model.parameters(), lr=self._lr)
        self._reward_loss = []

    def save(self, seed: int = 1, file_name: str = None):
        """Save reward model network and the reward model loss"""
        self._network.save(file_name + "/reward_model_{}".format(seed))
        np.save(file_name + "/reward_model_loss_{}".format(seed), self.get_current_loss())

    def load(self, seed: int = 1, file_name: str = None):
        if file_name is None:
            raise ValueError("The log directory is empty")
        self._network.load(file_name + "/reward_model_{}".format(seed))

    def report(self):
        return self._network.model

    def get_current_loss(self) -> np.ndarray:
        return np.array(self._reward_loss)
