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
                        'test/environment_reward': episode_return.mean()
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
                next_action = self._actor(next_obs)
                episode_loss += self._critic.update(obs, action, reward, term, next_obs, next_action)
                episode_return += reward
                if term or trunc:
                    break
                obs = next_obs

            wandb.log(
                {
                    'train/environment_reward': episode_return,
                    'train/loss_mdp': episode_loss
                },
                step=ep,
                commit=True
            )
            self._actor.update()

        wandb.finish()
        self._env.close()

    def test(self):
        episode_return = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                episode_return[ep] += reward
                if term or trunc:
                    break
                obs = next_obs
        return episode_return




class MonExperiment(Experiment):
    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        for ep in tqdm(range(self._training_episodes)):
            if ep % self._testing_frequency == 0:
                self._actor.eval()
                episode_return_true, episode_return_proxy, episode_return_cost = self.test()
                episode_return_true = episode_return_true.mean()
                episode_return_proxy = np.nanmean(episode_return_proxy)
                episode_return_cost = episode_return_cost.mean()
                self._actor.train()
                wandb.log(
                    {
                        'test/environment_reward': episode_return_true,
                        'test/received_reward': episode_return_proxy,
                        'test/monitor_reward': episode_return_cost
                    },
                    step=ep,
                    commit=False
                )

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return_true = 0.
            episode_return_proxy = 0.
            episode_return_cost = 0.
            episode_loss_mdp = 0.
            episode_loss_mon = 0.
            reward_seen = False
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                next_action = self._actor(next_obs)
                step_loss_mdp, step_loss_mon = \
                    self._critic.update(obs, action, reward, term, next_obs, next_action)

                episode_return_true += info['mdp_reward']
                episode_return_cost += reward['monitor']
                if not np.isnan(reward['mdp']):
                    reward_seen = True
                    episode_return_proxy += reward['mdp']
                if not np.isnan(step_loss_mdp):
                    episode_loss_mdp += step_loss_mdp
                if not np.isnan(step_loss_mon):
                    episode_loss_mon += step_loss_mon

                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy = np.nan
                    break

                obs = next_obs

            wandb.log(
                {
                    'train/environment_reward': episode_return_true,
                    'train/received_reward': episode_return_proxy,
                    'train/monitor_reward': episode_return_cost,
                    'train/loss_mdp': episode_loss_mdp,
                    'train/loss_mon': episode_loss_mon
                },
                step=ep,
                commit=True
            )
            self._actor.update()

        wandb.finish()
        self._env.close()

    def test(self):
        episode_return_true = np.zeros(self._testing_episodes)
        episode_return_proxy = np.zeros(self._testing_episodes)
        episode_return_cost = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            reward_seen = False
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            while True:
                action = self._actor(obs)
                next_obs, reward, term, trunc, info = self._env.step(action)
                episode_return_true[ep] += info['mdp_reward']
                episode_return_cost[ep] += reward['monitor']
                if not np.isnan(reward['mdp']):
                    reward_seen = True
                    episode_return_proxy[ep] += reward['mdp']
                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy[ep] = np.nan
                    break
                obs = next_obs

        return episode_return_true, episode_return_proxy, episode_return_cost
