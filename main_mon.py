import gymnasium as gym
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.actor import MonEpsilonGreedy
from src.critic import MonQDict, MonQTable
from src.experiment import MonExperiment
from src.wrappers import env_wrappers, monitor_wrappers


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg : DictConfig) -> None:
    wandb.init(
        group=cfg["exp"]["environment"]["id"],
        config=cfg["exp"],
        **cfg["wandb"],
    )

    env = gym.make(**cfg["exp"]["environment"])

    if 'MiniGrid' in cfg["exp"]["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        env = monitor_wrappers.BinaryMonitor(env, **configs["exp"]["monitor"])
        critic = MonQDict(env.observation_space, env.action_space, **configs["exp"]["critic"])

    else:
        env = monitor_wrappers.BinaryMonitor(env, **configs["exp"]["monitor"])
        critic = MonQTable(env.observation_space, env.action_space, **configs["exp"]["critic"])

    actor = MonEpsilonGreedy(critic, **configs["exp"]["actor"])

    experiment = MonExperiment(env, actor, critic, **configs["exp"]["experiment"])

    experiment.train()


if __name__ == "__main__":
    run()
