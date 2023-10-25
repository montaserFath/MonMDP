"""Replay buffer class to save data then use it to train a value function"""
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_size: tuple, action_size: tuple, max_size: int = int(1e6)):
        self.obs_size = obs_size
        self.action_size = action_size
        self.max_size = max_size
        self.buffer = None
        self.buffer_size = 0
        self.traj_size = int(2 * obs_size[0]) if len(obs_size) == 1 else int(2 * obs_size[0] * obs_size[1])  # Obs size
        self.traj_size += 2  # add action & reward

        # reset the buffer
        self.reset()

    def reset(self) -> None:
        self.buffer = {}
        self.buffer_size = 0

    # TODO: Add different sampling methods priority
    def sample(self, n_samples: int) -> np.ndarray:
        # check if the buffer is empty
        if not self.buffer:
            raise ValueError("The replay buffer is empty")

        # check if the number of sample is bigger than the buffer size
        if len(self.buffer.keys()) <= n_samples:
            raise ValueError("the number of sample is bigger than the buffer size")

        # random sampling
        samples_idx = np.random.randint(0, self.buffer_size, size=n_samples)
        # return dict(zip(samples_idx, [self.buffer[sample] for sample in samples_idx]))
        return np.array([self.buffer[sample] for sample in samples_idx]).reshape(n_samples, self.traj_size)

    # add single trajectory to the buffer
    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray) -> None:
        # if the buffer is full remove the first sample
        if len(self.buffer.keys()) == self.max_size:
            self.pop()
        new_sample = np.concatenate((obs.reshape(-1), action, reward, next_obs.reshape(-1)))
        self.buffer.update({self.buffer_size: new_sample})
        self.buffer_size += 1

    def size(self) -> int:
        return self.buffer_size

    # remove he first item in the buffer
    def pop(self) -> None:
        self.buffer.pop(0)
        self.buffer_size -= 1
