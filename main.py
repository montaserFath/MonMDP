import gymnasium as gym
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import config_parser, arg_parser
from src.actor import EpsilonGreedy
from src.critic import QTable, QDict
from src.experiment import Experiment
from src.wrappers import env_wrappers


@hydra.main(version_base=None)
def run(cfg : DictConfig) -> None:
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
    args = arg_parser()
    cfg = config_parser(args.config)

    wandb.init(
        entity="ualberta-bowling",
        project="QL demo",
        group=configs["environment"]["id"],
        mode=args.wandb_mode,
        config=cfg,
    )

    run(cfg)
