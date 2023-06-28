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
        group=cfg["environment"]["id"],
        config=cfg,
        **cfg["wandb"],
    )

    env = gym.make(**cfg["environment"])

    if 'MiniGrid' in cfg["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        env = getattr(monitor_wrappers, cfg["monitor"]["id"])(env, **cfg["monitor"])
        critic = MonQDict(env.observation_space, env.action_space, **cfg["agent"]["critic"])

    else:
        env = getattr(monitor_wrappers, cfg["monitor"]["id"])(env, **cfg["monitor"])
        critic = MonQTable(env.observation_space, env.action_space, **cfg["agent"]["critic"])

    actor = MonEpsilonGreedy(critic, **cfg["agent"]["actor"])

    experiment = MonExperiment(env, actor, critic, **cfg["experiment"])

    experiment.train()


if __name__ == "__main__":
    run()
