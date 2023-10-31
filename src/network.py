"""Neural Network class"""
import numpy as np
import torch
import torch.nn as nn


class NeuralNetwork:
    def __init__(
            self,
            obs_size: int,
            n_actions: int,
            lr: float = 0.001,
            gamma: float = 0.99,
            device: str = "cpu",
    ):
        self._obs_size = obs_size
        self._n_actions = n_actions
        self._lr = lr
        self._gamma = gamma
        self._device = device
        self.loss = nn.L1Loss()
        self.optimizer = None
        self.model = None
        self.reset()

    def train(self, train_data, n_epochs: int, batch_size: int):
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            for batch in range(0, train_data.shape[0], batch_size):
                batch_obs =
                batch_reward =
        NotImplemented

    def eval(self, test_obs):
        if len(test_obs.shape) == 1:
            test_obs = test_obs.reshape(1, test_obs.shape[1])
        # with torch.no_grad():
        return self.model(self.numpy_to_torch(test_obs)).detach().cpu().numpy()

    # convert sample to pytorch format
    def numpy_to_torch(self, obs: np.ndarray):
        if obs.shape[1] != self._obs_size:
            raise ValueError("Observations size does not match observation shape")
        return torch.from_numpy(obs)

    def process_batch(self, batch: np.ndarray):
        terminal_idx = np.where()
        expected_q = torch.zeros(batch.shape[0])

    def reset(self):
        self.model = nn.Sequential(
            nn.Linear(self._obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self._n_actions),
        ).to(self._device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)

    def save(self, log_dir: str = None):
        if log_dir is None:
            raise ValueError("The log directory is empty")
        torch.save(self.model, log_dir)

    def load(self, log_dir: str = None):
        if log_dir is None:
            raise ValueError("The log directory is empty")
        self.model = torch.load(log_dir)
