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
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
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
            batch_inds = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.buffer_current_size
            ) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.buffer_current_size, size=batch_size)

        data = (
            {"mdp": self.mdp_observations[batch_inds], "monitor": self.mon_observations[batch_inds]},
            self.actions[batch_inds],
            {"mdp": self.mdp_next_observations[batch_inds], "monitor": self.mon_next_observations[batch_inds]},
            (self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).reshape(-1, 1),
            {"mdp": self.rewards[batch_inds, 0], "monitor": self.rewards[batch_inds, 1]},
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
            self.timeouts[self.buffer_current_size] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

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
        return self.buffer
        # self.buffer = np.load(log_dir + "/replay_buffer.npy")


class TorchReplayMemory:
    """Replay memory of (obs, action, next_obs, reward) transitions.

    obs_dtype: store the observation windows (obs["mdp"], next_obs["mdp"]) in this dtype, e.g. "float16", instead of
    the env's float64 (None keeps them as they are). `process_batch` casts the windows to float32 anyway, so this only
    changes what is kept in memory; float16 is exact for the values 0, 0.5 and 1 and rounds others (e.g. 1/3) by about
    1e-4. A window that would change by more than `obs_tolerance` (e.g. an overflow) raises instead of being stored.
    """

    def __init__(self, max_size: int, device: str = None, obs_dtype=None, obs_tolerance: float = 1e-3):
        self.max_size = max_size
        self.device = device
        self.obs_dtype = None if obs_dtype is None else np.dtype(obs_dtype)
        self.obs_tolerance = obs_tolerance
        self._last_source = self._last_cast = None
        self.buffer_size = 0
        self.memory = deque([], maxlen=self.max_size)
        self.transition = namedtuple("Transition", ("obs", "action", "next_obs", "reward"))

    def _cast_obs(self, obs: dict) -> dict:
        """obs dict -> dict whose 'mdp' window is stored in obs_dtype. A window is cast once: the training loop reuses
        next_obs as the next step's obs, so the cast array is shared by both transitions and stored once."""
        window = obs["mdp"]
        if window is None:  # terminal next state
            return obs
        if window is not self._last_source:
            with np.errstate(over="ignore", invalid="ignore"):  # an overflow / NaN is caught by the check below
                cast = np.asarray(window).astype(self.obs_dtype)
                error = np.abs(cast.astype(np.float64) - window).max()
            if not error <= self.obs_tolerance:  # also rejects NaN
                raise ValueError(f"observation window does not fit in {self.obs_dtype} (tolerance {self.obs_tolerance})")
            self._last_source, self._last_cast = window, cast
        return {**obs, "mdp": self._last_cast}

    def push(self, *args):
        """Save a transition"""
        if self.obs_dtype is not None:
            obs, action, next_obs, reward = args
            args = (self._cast_obs(obs), action, self._cast_obs(next_obs), reward)
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

    def save(self, log_dir: str, seed: int):
        np.save(log_dir + "/replay_buffer_{}.npy".format(seed), self.memory)

    def load(self, log_dir: str, seed: int = 1):
        self.memory = np.load(log_dir + "replay_buffer_{}.npy".format(seed), allow_pickle=True)[()]
        self.memory = deque([list(i) for i in self.memory])
        self.buffer_size = len(self.memory)

    def process_batch(self, batch, device: str):
        batch = self.transition(*zip(*batch))
        real_reward_mask = [not np.isnan(p) for p in [r["mdp"] for r in batch.reward]]
        if len(real_reward_mask) == 0:
            raise ValueError("All nan")
        real_reward_idx = [i for i, x in enumerate(real_reward_mask) if x]
        non_final_mask = torch.tensor(tuple(map(lambda s: s["mdp"] is not None, batch.next_obs)), dtype=torch.bool)
        non_final_next_states = torch.tensor(
            np.array([s["mdp"] for s in batch.next_obs if s["mdp"] is not None]),
            device=device,
            dtype=torch.float,
        )

        non_final_next_monitor_states = torch.tensor(
            np.array([s["monitor"] for s in batch.next_obs if s["monitor"] is not None]),
            device=device,
            dtype=torch.float,
        )

        mdp_obs = torch.tensor(np.array([state["mdp"] for state in batch.obs]), device=device, dtype=torch.float)
        mon_obs = torch.tensor(
            np.array([state["monitor"] for state in batch.obs]).astype(np.float32), device=device, dtype=torch.float
        )

        mdp_reward = torch.tensor(np.array([r["mdp"] for r in batch.reward]), device=device, dtype=torch.float)
        mon_reward = torch.tensor(np.array([r["monitor"] for r in batch.reward]), device=device, dtype=torch.float)
        # logging only, never used for training: the env's true reward (NaN if the transition was stored without it)
        # and the number of reward changes at the time of the transition (-1 for a stationary env)
        true_reward = torch.tensor(
            np.array([r.get("true", np.nan) for r in batch.reward]), device=device, dtype=torch.float
        )
        change_idx = torch.tensor(np.array([r.get("change_idx", -1) for r in batch.reward]), device=device,
                                  dtype=torch.float)

        mdp_action = torch.tensor(np.array([a["mdp"] for a in batch.action]), device=device)
        mon_action = torch.tensor(np.array([a["monitor"] for a in batch.action]), device=device)
        process_batch = {
            "real_reward_idx": real_reward_idx,
            "mdp_obs": mdp_obs,
            "mon_obs": mon_obs.unsqueeze(1),
            "non_final_mask": non_final_mask,
            "non_final_next_states": non_final_next_states,
            "non_final_next_monitor_states": non_final_next_monitor_states.unsqueeze(1),
            "mdp_reward": mdp_reward.unsqueeze(1),
            "mon_reward": mon_reward.unsqueeze(1),
            "mdp_action": mdp_action.unsqueeze(1),
            "mon_action": mon_action.unsqueeze(1),
            "true_reward": true_reward.unsqueeze(1),  # logging only
            "change_idx": change_idx.unsqueeze(1),  # logging only
        }
        return process_batch
