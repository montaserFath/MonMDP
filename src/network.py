"""Neural Network class"""
from abc import abstractmethod, ABC
import numpy as np
import torch


class NeuralNetwork:
    def __init__(
        self,
        obs_size: tuple,
        n_actions: int,
        lr: float = 0.001,
        device: str = None,  # "cpu"
    ):
        self._obs_size = obs_size
        self._n_actions = n_actions
        self._lr = lr
        self._device = device
        self.loss_fun = torch.nn.MSELoss()
        # self.optimizer = None
        self.model = None
        # self.reset()

    def __call__(self, data):
        return self.model(data)

    @abstractmethod
    def train(self, train_data, n_epochs: int, batch_size: int):
        # for epoch in range(n_epochs):
        #     epoch_loss = 0
        #     n_batches = 0
        #     for batch in range(0, train_data.shape[0], batch_size):
        #         batch_obs =
        #         batch_obs_reward =
        return

    def eval(self, test: np.ndarray):
        # if its 1d tensor/array convert it to 2d tensor/array
        if len(test.shape) == 1:
            test = test.reshape(1, test.shape[1])
        # if numpy array convert it to tensor
        if isinstance(test, np.ndarray):
            test = torch.from_numpy(test)
        # with torch.no_grad():
        return self.model(test).detach().cpu().numpy()

    def init_network(self):
        input_size = 1
        for item in self._obs_size:
            input_size *= item
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self._n_actions),
        ).to(self._device)

    def save(self, log_dir: str = None):
        if log_dir is None:
            raise ValueError("The log directory is empty")
        torch.save(self.model, log_dir)

    def load(self, log_dir: str = None):
        if log_dir is None:
            raise ValueError("The log directory is empty")
        self.model = torch.load(log_dir)

    def parameters(self):
        pass

    @property
    def device(self):
        return self._device


class CNN(NeuralNetwork):
    def __init__(
        self,
        obs_size: tuple,
        n_actions: int,
        lr: float = 0.001,
        kernel_size_0: int = 5,
        kernel_size_1: int = 3,
        add_monitor_obs: bool = False,
        output_channels: int = 64,
        device: str = None,
        features_dim: int = 512,
    ):
        """
        :param obs_size: tuple
        :param n_actions: (int) number of actions
        :param kernel_size_0: (int) kernel size for the first convolutional layer
        :param kernel_size_1: (int) kernel size for the second convolutional layer
        :param add_monitor_obs: (bool) add monitor observation
        :param output_channels: (int) number of output channels for the second convolutional layer
        :param lr: (float) learning rate for the optimizer
        :param device: (str) device name "cpu" or "mps:0" for mac "cuda:0" for cuda
        :param features_dim: (int) Number of features extracted. This corresponds to the number of unit for the last layer.
        """
        NeuralNetwork.__init__(self, obs_size, n_actions, lr, device)
        self._device = device
        # We assume CxHxW images (channels first)
        if len(self._obs_size) != 3:
            raise ValueError("The observation size should be CxHxW")
        self.features_dim = features_dim
        self.kernel_size_0 = kernel_size_0
        self.kernel_size_1 = kernel_size_1
        self.add_monitor_obs = add_monitor_obs
        self.output_channels = output_channels
        if self.kernel_size_0 >= self._obs_size[1]:
            raise ValueError("The kernel size of the first convolutional layer is larger than input size")

        if self.kernel_size_1 >= self._obs_size[1] - self.kernel_size_0 + 1:
            raise ValueError("The kernel size of the second convolutional layer is larger than input size")

        self.init_network()

    def forward(self, obs, monitor_obs = None) -> torch.Tensor:
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self._device).unsqueeze(0)
        if monitor_obs is None:
            return self.model(obs).to(self._device)
        if isinstance(monitor_obs, int):
            monitor_obs = torch.tensor([monitor_obs], dtype=torch.float, device=self._device).unsqueeze(0)
        inter_obs = torch.concat((self.model[:5](obs), monitor_obs), 1)
        return self.model[5:](inter_obs)

    def init_network(self):
        ch_out = self._obs_size[1] - self.kernel_size_0 + 1 - self.kernel_size_1 + 1
        n_flatten = int(ch_out**2 * self.output_channels) + (1 if self.add_monitor_obs else 0)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(self._obs_size[0], 32, self.kernel_size_0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, self.output_channels, self.kernel_size_1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(n_flatten, self.features_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.features_dim, self._n_actions),
        ).to(self._device)

    def device(self):
        return self._device
