"""Train/evaluate/Plot in Plants watering environments"""
import gymnasium as gym
import os
import hydra
from src.utils import dict_to_id

import wandb
from omegaconf import DictConfig, OmegaConf

from src.actor import MonEpsilonGreedyOneAction
from src.critic import MonQCNN
from src.experiment import MonExperiment
from src.wrappers.env_wrappers import WallObs, WindowViewObs
from src.wrappers.monitor_wrappers import BinaryMonitor

EVAL = False
LOG_DIR = "models/9_9/Plants/reward_model/env_0.0/eps_1.0/q_lr_1.0/reward_lr_1.0/"


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_monitor(cfg: DictConfig) -> None:
    """Run env"""
    wandb.init(
        group="gym_monitor/Plants-Watering-v1" + "\\" + dict_to_id(cfg.monitor),
        config=OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        ),
        settings=wandb.Settings(start_method="thread"),
        **cfg.wandb,
    )

    env = gym.make(
        cfg.environment.id,
        grid_size=cfg.environment.grid_size,
        n_plants=cfg.environment.n_plants,
        plants_dryness_prob=cfg.environment.plants_dryness_prob,
        dry_difference=cfg.environment.dry_difference,
        agent_start_pos=cfg.environment.agent_start_pos,
        max_episode_steps=cfg.environment.max_episode_steps,
    )
    env = WallObs(env, grid_size=cfg.environment.grid_size, n_walls=cfg.environment.n_walls)
    env = WindowViewObs(env, window_size=cfg.environment.window_size)
    env = BinaryMonitor(env, full_monitor=False, **cfg.monitor)

    q_lr, reward_lr, eps = cfg.agent.critic.lr, cfg.agent.critic.reward_model.lr, cfg.agent.actor.init_eps
    dry, window_size = cfg.environment.plants_dryness_prob, cfg.environment.window_size
    eps = eps if cfg.agent.actor.init_eps == cfg.agent.actor.min_eps else "decay"
    train_dir = "general_models/Plants/" + "/" + str(cfg.agent.critic.strategy) + "/6_6/dry_{}/".format(dry)
    train_dir += "eps_{}/window_{}/q_lr_{}/reward_lr_{}/".format(eps, window_size, q_lr, reward_lr)
    os.makedirs(train_dir, exist_ok=True)

    critic = MonQCNN(
        cfg.environment.id, env.observation_space, env.action_space, dir_name=train_dir, **cfg.agent.critic,
    )
    actor = MonEpsilonGreedyOneAction(critic, train=not EVAL, **cfg.agent.actor)
    experiment = MonExperiment(
        env, actor, critic, log_dir=LOG_DIR if EVAL else train_dir, replay_buffer=True, **cfg.experiment,
    )
    experiment.train()


if __name__ == "__main__":
    run_monitor()
