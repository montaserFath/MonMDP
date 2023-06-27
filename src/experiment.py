import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

from src.actor import Actor
from src.critic import Critic
from src.utils import set_rng_seed, cantor_pairing


class Experiment():
    def __init__(self, env: gym.Env, actor: Actor, critic: Critic,
                 training_episodes, testing_episodes, testing_frequency, rng_seed):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._training_episodes = training_episodes
        self._testing_episodes = testing_episodes
        self._testing_frequency = testing_frequency
        self._rng_seed = rng_seed

    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        for ep in tqdm(range(self._training_episodes)):
            if ep % self._testing_frequency == 0:
                self._actor.eval()
                episode_return = self.test()
                self._actor.train()
                wandb.log(
                    {
                        'test/return_true': np.nanmean(episode_return)
                    },
                    step=ep,
                    commit=False
                )

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return = 0.
            episode_loss = 0.
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                episode_return += reward
                episode_loss += self._critic.update(obs, action, reward, term, next_obs)
                if term or trunc:
                    break
                obs = next_obs

            wandb.log(
                {
                    'train/return_true': episode_return,
                    'train/loss': episode_loss
                },
                step=ep,
                commit=True
            )
            self._actor.update()

        wandb.finish()
        self._env.close()

    def test(self):
        episode_returns = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                episode_returns[ep] += reward
                if term or trunc:
                    break
                obs = next_obs
        return episode_returns




class MonExperiment(Experiment):
    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        for ep in tqdm(range(self._training_episodes)):
            if ep % self._testing_frequency == 0:
                self._actor.eval()
                episode_returns_true, episode_returns_proxy, episode_returns_cost = self.test()
                self._actor.train()
                wandb.log(
                    {
                        'test/return_true': np.nanmean(episode_returns_true),
                        'test/return_proxy': np.nanmean(episode_returns_proxy),
                        'test/return_cost': np.nanmean(episode_returns_cost)
                    },
                    step=ep,
                    commit=False
                )

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return_true = 0.
            episode_return_proxy = 0.
            episode_return_cost = 0.
            episode_loss = [0., 0.]
            reward_seen = False
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                episode_return_true += info['mdp_reward']
                episode_return_cost += reward['monitor']
                if reward['mdp'] is not np.nan:
                    reward_seen = True
                    episode_return_proxy += reward['mdp']

                step_loss = self._critic.update(obs, action, reward, term, next_obs)
                if step_loss[0] is not np.nan:
                    episode_loss[0] += step_loss[0]
                if step_loss[1] is not np.nan:
                    episode_loss[1] += step_loss[1]

                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy = []
                    break
                obs = next_obs

            wandb.log(
                {
                    'train/return_true': episode_return_true,
                    'train/return_proxy': episode_return_proxy,
                    'train/return_cost': episode_return_cost,
                    'train/loss': episode_loss[0],
                    'train/loss_mon': episode_loss[1]
                },
                step=ep,
                commit=True
            )
            self._actor.update()

        wandb.finish()
        self._env.close()

    def test(self):
        episode_returns_true = np.zeros(self._testing_episodes)
        episode_returns_proxy = np.zeros(self._testing_episodes)
        episode_returns_cost = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            reward_seen = False
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                episode_returns_true[ep] += info['mdp_reward']
                episode_returns_cost[ep] += reward['monitor']
                if reward['mdp'] is not np.nan:
                    reward_seen = True
                    episode_returns_proxy[ep] += reward['mdp']
                if term or trunc:
                    if not reward_seen:
                        episode_returns_proxy[ep] = np.nan
                    break

                obs = next_obs
        return episode_returns_true, episode_returns_proxy, episode_returns_cost
