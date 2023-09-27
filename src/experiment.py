import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

from src.actor import Actor
from src.critic import Critic
from src.utils import set_rng_seed, cantor_pairing


class Experiment:
    def __init__(self, env: gym.Env, actor: Actor, critic: Critic,
                 training_timesteps, testing_timesteps, testing_frequency, rng_seed, log_dir: str, save_log: bool = False):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._training_timesteps = training_timesteps
        self._testing_timesteps = testing_timesteps
        self._testing_frequency = testing_frequency
        self._rng_seed = rng_seed
        self._log_dir = log_dir
        self._save_train_log = save_log

    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()
        joint_reward = {}
        total_timesteps = 0
        episode = 0
        while total_timesteps < self._training_timesteps:
            if episode % self._testing_frequency == 1:
                self._actor.eval()
                episode_return = self.test()
                self._actor.train()
                wandb.log(
                    {
                        'test/environment_reward': episode_return.mean()
                    },
                    step=episode,
                    commit=False
                )

            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return = 0.
            episode_loss = 0.
            next_action = None
            ep_joint_reward = []
            episode_timesteps = 0
            while True:
                episode_timesteps += 1
                action = self._actor(obs) if next_action is None else next_action
                next_obs, reward, term, trunc, _ = self._env.step(action)
                if self._critic._on_policy:
                    next_action = self._actor(next_obs)
                episode_loss += self._critic.update(obs, action, reward, term, next_obs, next_action)
                ep_joint_reward.append(reward)
                episode_return += reward
                if term or trunc:
                    break
                obs = next_obs
                self._actor.update()
            joint_reward[episode_timesteps] = ep_joint_reward
            total_timesteps += episode_timesteps
            episode += 1
            wandb.log(
                {
                    'train/environment_reward': episode_return,
                    'train/loss_mdp': episode_loss
                },
                step=episode,
                commit=True
            )
        if self._save_train_log:
            np.save(self._log_dir + "/training_joint_reward_{}.npy".format(self._rng_seed), joint_reward)
        # save Q-table as numpy array
        self._critic.save(seed=self._rng_seed)
        wandb.finish()
        self._env.close()

    def test(self, render: bool = False, save_results: bool = False):
        episode_return = []
        total_timesteps = 0
        episode = 0
        while total_timesteps < self._testing_timesteps:
            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_timesteps = 0
            total_reward = 0
            while True:
                episode_timesteps += 1
                if render:
                    self._env.render()
                action = self._actor(obs)
                next_obs, reward, term, trunc, _ = self._env.step(action)
                total_reward += reward
                if term or trunc:
                    break
                obs = next_obs
            episode_return.append(total_reward)
            episode += 1
            total_timesteps += episode_timesteps
        return np.array(episode_return)


class MonExperiment(Experiment):
    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()
        joint_reward = {}
        eval_joint_reward = []
        eval_count = 0
        total_timesteps = 0
        episode = 0
        while total_timesteps < self._training_timesteps:
            if episode > 0 and episode % self._testing_frequency == 0:
                self._actor.eval()
                ep_return_true, ep_return_proxy, ep_return_cost, ep_monitor_action, ep_length, _ = self.test()
                eval_joint_reward.append(ep_return_true + ep_return_cost)
                episode_return_true = ep_return_true.mean()
                episode_return_proxy = np.nanmean(ep_return_proxy)
                episode_return_cost = ep_return_cost.mean()
                self._actor.train()
                wandb.log(
                    {
                        'test/environment_reward': episode_return_true,
                        'test/received_reward': episode_return_proxy,
                        'test/monitor_reward': episode_return_cost,
                        "test/monitor_action": np.mean(ep_monitor_action),
                        "test/number_of_timesteps": np.mean(ep_length),
                        "test/joint_reward": episode_return_true + episode_return_cost,
                    },
                    step=episode,
                    commit=False
                )
                eval_count += 1

            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_return_true = 0.
            episode_return_proxy = 0.
            episode_return_cost = 0.
            episode_loss_mdp = 0.
            episode_loss_mon = 0.
            reward_seen = False
            episode_monitor_action_count, episode_timesteps = 0, 0
            next_action = None
            ep_joint_reward = []
            while True:
                episode_timesteps += 1
                action = self._actor(obs) if next_action is None else next_action

                if action["monitor"] == 1:
                    episode_monitor_action_count += 1
                next_obs, reward, term, trunc, info = self._env.step(action)
                if self._critic._on_policy:
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

                ep_joint_reward.append(info['mdp_reward'] + reward['monitor'])
                self._actor.update()
                if term or trunc:
                    if not reward_seen:
                        episode_return_proxy = np.nan
                    break

                obs = next_obs
            joint_reward.update({episode: ep_joint_reward})
            total_timesteps += episode_timesteps
            wandb.log(
                {
                    'train/environment_reward': episode_return_true,
                    'train/received_reward': episode_return_proxy,
                    'train/monitor_reward': episode_return_cost,
                    'train/loss_mdp': episode_loss_mdp,
                    'train/loss_mon': episode_loss_mon,
                    "train/monitor_action": episode_monitor_action_count,
                    "train/number_of_timesteps": episode_timesteps,
                    "train/joint_reward": episode_return_true + episode_return_cost,
                },
                step=episode,
                commit=True
            )
            episode += 1
        # save Q-table as numpy array
        self._critic.save(seed=self._rng_seed)
        if self._save_train_log:
            np.save(self._log_dir + "/training_joint_reward_{}.npy".format(self._rng_seed), joint_reward)
            np.save(
                self._log_dir + "/evaluation_joint_reward_{}.npy".format(self._rng_seed), np.array(eval_joint_reward),
            )
        wandb.finish()
        self._env.close()

    def test(self, render: bool = False, save_results: bool = False):
        episode_return_true = []
        episode_return_proxy = []
        episode_return_cost = []
        episode_monitor_action = []
        episode_length = []
        trajectories = {}
        total_timesteps = 0
        episode = 0
        while total_timesteps < self._testing_timesteps:
            reward_seen = False
            ep_seed = cantor_pairing(self._rng_seed, episode)
            obs, _ = self._env.reset(seed=ep_seed)
            episode_monitor_action_count, episode_timesteps = 0, 0
            ep_states, ep_actions = [], []
            return_true, return_proxy, return_cost = 0, 0, 0
            while True:
                ep_states.append([obs["mdp"].item(), obs["monitor"]])
                episode_timesteps += 1
                if render:
                    self._env.render()
                action = self._actor(obs)
                ep_actions.append([action["mdp"], action["monitor"]])
                if action["monitor"] == 1:
                    episode_monitor_action_count += 1
                next_obs, reward, term, trunc, info = self._env.step(action)
                return_true += info['mdp_reward']
                return_cost += reward['monitor']
                if not np.isnan(reward['mdp']):
                    reward_seen = True
                    return_proxy += reward['mdp']
                if term or trunc:
                    if not reward_seen:
                        return_proxy = np.nan
                    break
                obs = next_obs
            episode_return_true.append(return_true)
            episode_return_cost.append(return_cost)
            episode_return_proxy.append(return_proxy)
            episode_monitor_action.append(episode_monitor_action_count)
            episode_length.append(episode_timesteps)
            total_timesteps += episode_timesteps
            trajectories[episode] = {
                "states": np.array(ep_states),
                "actions": np.array(ep_actions),
                "environment_reward": np.array(episode_return_true),
                "received_reward": np.array(episode_return_proxy),
                "monitor_reward": np.array(episode_return_cost),
                "joint_reward": np.array(episode_return_true) + np.array(episode_return_cost),
            }
            episode += 1
        if save_results:
            np.save(self._log_dir + "/trajectories.npy", trajectories)
        return (
            np.array(episode_return_true),
            np.array(episode_return_proxy),
            np.array(episode_return_cost),
            np.array(episode_monitor_action),
            np.array(episode_length),
            trajectories,
        )
