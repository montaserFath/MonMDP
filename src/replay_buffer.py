"""Replay buffer class to save data then use it to train a value function"""
import pickle
import random
from collections import namedtuple, deque
from typing import Union, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from src.policy_analysis import ind_to_action, get_action_ind


class OldReplayBuffer:
    def __init__(self, obs_size: tuple, action_size: tuple, max_size: int = int(1e6)):
        self.obs_size = obs_size
        self.action_size = action_size
        self.max_size = max_size
        self.buffer = None
        self.buffer_size = 0
        self.transition = None
        self.traj_size = int(2 * obs_size[0]) if len(obs_size) == 1 else int(2 * obs_size[0] * obs_size[1])  # Obs size
        self.traj_size += 2  # add action & reward

        # reset the buffer
        self.reset()

    def reset(self) -> None:
        self.buffer = deque([], maxlen=self.max_size)
        self.buffer_size = 0
        self.transition = namedtuple("Transition", ("obs", "action", "next_obs", "reward"))

    # TODO: Add different sampling methods priority
    def sample(self, n_samples: int) -> np.ndarray:
        # check if the buffer is empty
        if not self.buffer:
            raise ValueError("The replay buffer is empty")

        # check if the number of sample is bigger than the buffer size
        if self.buffer_size <= n_samples:
            raise ValueError("the number of sample is bigger than the buffer size")
        return random.sample(list(self.buffer), n_samples)

        # random sampling
        # samples_idx = np.random.randint(0, self.buffer_size, size=n_samples)
        # # return dict(zip(samples_idx, [self.buffer[sample] for sample in samples_idx]))
        # return np.array([self.buffer[sample] for sample in samples_idx]).reshape(n_samples, self.traj_size)

    # add single trajectory to the buffer
    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, args) -> None:
        # if the buffer is full remove the first sample
        if len(self.buffer.keys()) == self.max_size:
            self.pop()
        # termination state
        # if next_obs is None:
        #     next_obs = np.array([None] * obs.shape[0])
        # sample: obs, action, reward, next obs
        # new_sample = np.concatenate((obs.reshape(-1), action, reward, next_obs.reshape(-1)))
        # self.buffer.update({self.buffer_size: new_sample})
        self.buffer.append(self.transition(*args))
        self.buffer_size += 1

    def size(self) -> int:
        # return len(self.buffer)
        return self.buffer_size

    # remove the first item in the buffer
    def pop(self) -> None:
        self.buffer.pop()
        self.buffer_size -= 1

    def save(self, log_dir: str = None) -> None:
        if log_dir is None:
            raise ValueError("Empty log dir")
        pickle.dump(self.buffer, open(log_dir + "/replay_buffer.pkl", "wb"))
        # np.save(self.buffer, log_dir + "/replay_buffer.npy")

    def load(self, log_dir: str = None) -> None:
        if log_dir is None:
            raise ValueError("Empty log dir")
        self.buffer = pickle.load(open(log_dir + "/replay_buffer.npy", "rb"))
        # self.buffer = np.load(log_dir + "/replay_buffer.npy")


class MonReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            buffer_size: int,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            device: Union[torch.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        """
        Replay buffer for Monitor MDP
        """
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)
        self.mdp_obs_size = observation_space["mdp"].shape
        self.mdp_n_actions = action_space["mdp"].n
        self.buffer_size = buffer_size
        self.buffer = None
        self.device = device
        self.buffer_current_size = 0
        self.transition = None
        self.traj_size = 1
        for size in self.obs_size:
            self.traj_size *= size
        self.traj_size *= 2  # state & next state
        self.traj_size += 2  # add action & reward

        # reset the buffer
        self.reset()

    def reset(self) -> None:
        """reset the replay buffer with the correct shapes"""
        self.mdp_observations = np.zeros((self.buffer_size, *self.mdp_obs_size))
        self.mon_observations = np.zeros(self.buffer_size)
        self.mdp_next_observations = np.zeros((self.buffer_size, *self.mdp_obs_size))
        self.mon_next_observations = np.zeros(self.buffer_size)
        self.actions = np.zeros((self.buffer_size, self.mdp_n_actions * 2))
        self.rewards = np.zeros((self.buffer_size, 2))
        self.dones = np.zeros(self.buffer_size)
        self.timeouts = np.zeros(self.buffer_size)

    # TODO: Add different sampling methods priority
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.buffer_current_size) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.buffer_current_size, size=batch_size)

        data = (
            {"mdp": self.mdp_observations[batch_inds], "monitor": self.mon_observations[batch_inds]},
            self.actions[batch_inds],
            {"mdp": self.mdp_next_observations[batch_inds], "monitor": self.mon_next_observations[batch_inds]},
            (self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).reshape(-1, 1),
            {"mdp": self.rewards[batch_inds, 0], "monitor": self.rewards[batch_inds, 1]}
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def add(self, obs: dict, action: dict, reward: dict, next_obs: dict, done: bool, infos: dict) -> None:
        self.mdp_observations[self.buffer_current_size] = obs["mdp"]
        self.mon_observations[self.buffer_current_size] = obs["monitor"]

        if next_obs["mdp"] is not None:
            self.mdp_next_observations[self.buffer_current_size] = next_obs["mdp"]
            self.mon_next_observations[self.buffer_current_size] = next_obs["monitor"]

        self.actions[self.buffer_current_size] = get_action_ind(action)

        # first colum is the received reward and the second colum is the monitored reward
        self.rewards[self.buffer_current_size] = [reward["mdp"], reward["monitor"]]
        self.dones[self.buffer_current_size] = done
        if self.handle_timeout_termination:
            self.timeouts[self.buffer_current_size] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.buffer_current_size += 1
        if self.buffer_current_size == self.buffer_size:
            self.full = True
            self.buffer_current_size = 0

    def size(self) -> int:
        # return len(self.buffer)
        return self.buffer_current_size

    # remove the first item in the buffer
    def pop(self) -> None:
        self.buffer_current_size -= 1

    def save(self, log_dir: str = None) -> None:
        if log_dir is None:
            raise ValueError("Empty log dir")
        pickle.dump(self.buffer, open(log_dir + "/replay_buffer.pkl", "wb"))
        # np.save(self.buffer, log_dir + "/replay_buffer.npy")

    def load(self, log_dir: str = None) -> None:
        if log_dir is None:
            raise ValueError("Empty log dir")
        self.buffer = pickle.load(open(log_dir + "/replay_buffer.npy", "rb"))
        # self.buffer = np.load(log_dir + "/replay_buffer.npy")


class TorchReplayMemory:

    def __init__(self, max_size: int, device: str = None):
        self.max_size = max_size
        self.device = device
        self.buffer_size = 0
        self.memory = deque([], maxlen=self.max_size)
        self.transition = namedtuple("Transition", ("obs", "action", "next_obs", "reward"))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))
        self.buffer_size += 1

    def sample(self, n_samples):
        # check if the buffer is empty
        if self.buffer_size == 0:
            raise ValueError("The replay buffer is empty")

        # check if the number of sample is bigger than the buffer size
        if self.buffer_size <= n_samples:
            raise ValueError("the number of sample is bigger than the buffer size")
        return random.sample(self.memory, n_samples)

    def __len__(self):
        return len(self.memory)

    def get_transition(self):
        return self.transition

    def save(self, log_dir: str):
        np.save(log_dir + "/replay_buffer.npy", self.memory)

    def load(self, log_dir: str):
        self.memory = np.load(log_dir + "replay_buffer.npy", allow_pickle=True)[()]

    def process_batch(self, batch, device: str):
        batch = self.transition(*zip(*batch))
        non_nan_mask = [not np.isnan(p) for p in [r["mdp"] for r in batch.reward]]
        if len(non_nan_mask) == 0:
            raise ValueError("All nan")
        non_nan_idx = [i for i, x in enumerate(non_nan_mask) if x]
        mdp_obs = torch.tensor([state["mdp"] for state in batch.obs], device=device, dtype=torch.float)[non_nan_idx]
        mon_obs = torch.tensor([state["monitor"] for state in batch.obs], device=device, dtype=torch.float)[non_nan_idx]

        non_final_mask = torch.tensor(tuple(map(lambda s: s["mdp"] is not None, batch.next_obs)), dtype=torch.bool)
        non_final_next_states = torch.tensor(
            [s["mdp"] for s in batch.next_obs if s["mdp"] is not None], device=device, dtype=torch.float,
        )

        mdp_reward = torch.tensor([reward["mdp"] for reward in batch.reward], device=device, dtype=torch.float)[non_nan_idx]
        mon_reward = torch.tensor([reward["monitor"] for reward in batch.reward], device=device, dtype=torch.float)[non_nan_idx]

        mdp_action = torch.tensor([a["mdp"] for a in batch.action], device=device)[non_nan_idx]
        mon_action = torch.tensor([a["monitor"] for a in batch.action], device=device)[non_nan_idx]

        return (
            mdp_obs,
            mon_obs.unsqueeze(1),
            non_final_mask,
            non_final_next_states,
            mdp_reward.unsqueeze(1),
            mon_reward.unsqueeze(1),
            mdp_action.unsqueeze(1),
            mon_action.unsqueeze(1),
        )
