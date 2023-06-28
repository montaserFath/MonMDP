import gymnasium as gym
import wandb
import hydra
from omegaconf import DictConfig

from src.actor import EpsilonGreedy
from src.critic import QTable, QDict
from src.experiment import Experiment
from src.wrappers import env_wrappers


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
        critic = QDict(env.observation_space, env.action_space, **cfg["agent"]["critic"])

    else:
        critic = QTable(env.observation_space, env.action_space, **cfg["agent"]["critic"])

    actor = EpsilonGreedy(critic, **cfg["agent"]["actor"])

    experiment = Experiment(env, actor, critic, **cfg["experiment"])

    experiment.train()


if __name__ == "__main__":
    run()
