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
        group=cfg["exp"]["environment"]["id"],
        config=cfg["exp"],
        name=f"seed: {cfg['exp']['experiment']['rng_seed']}",
        **cfg["wandb"],
    )

    env = gym.make(**cfg["exp"]["environment"])

    if 'MiniGrid' in cfg["exp"]["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        critic = QDict(env.observation_space, env.action_space, **cfg["exp"]["critic"])

    else:
        critic = QTable(env.observation_space, env.action_space, **cfg["exp"]["critic"])

    actor = EpsilonGreedy(critic, **cfg["exp"]["actor"])

    experiment = Experiment(env, actor, critic, **cfg["exp"]["experiment"])

    experiment.train()


if __name__ == "__main__":
    run()
