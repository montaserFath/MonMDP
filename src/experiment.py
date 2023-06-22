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
        wandb.init(
            project='QL demo',
            mode='online',
        )

        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        for ep in tqdm(range(self._training_episodes)):
            if ep % self._testing_frequency == 0:
                self._actor.eval()
                episode_return = self.test()
                self._actor.train()
                wandb.log(
                    {'test/return': episode_return.mean()},
                    step=ep,
                    commit=False
                )

            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return = 0.
            episode_loss = 0.
            steps = 0
            while True:
                steps += 1
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                episode_return += reward
                episode_loss += self._critic.update(obs, action, reward, term, next_obs)
                if term or trunc:
                    break
                obs = next_obs

            wandb.log(
                {'train/return': episode_return, 'train/loss': episode_loss},
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
