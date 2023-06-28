import gymnasium as gym
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.actor import EpsilonGreedy
from src.critic import QTable, QDict
from src.experiment import Experiment
from src.wrappers import env_wrappers


@hydra.main(version_base=None, config_path="configs", config_name="taxi_ql")
def run(cfg : DictConfig, wandb_mode : str = None) -> None:
    # wandb.init(
    #     entity="ualberta-bowling",
    #     group=cfg["environment"]["id"],
    #     project="monitor parisi",
    #     mode=args.wandb_mode,
    #     config=cfg,
    # )

    print(wandb_mode)
    return

    env = gym.make(**cfg["environment"])
    if 'MiniGrid' in cfg["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        critic = QDict(env.observation_space, env.action_space, **cfg["critic"])
    else:
        critic = QTable(env.observation_space, env.action_space, **cfg["critic"])
    actor = EpsilonGreedy(critic, **cfg["actor"])
    experiment = Experiment(env, actor, critic, **cfg["experiment"])

    experiment.train()


if __name__ == "__main__":
    run()
