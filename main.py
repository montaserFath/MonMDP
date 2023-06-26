import gymnasium as gym
import wandb

from src.utils import config_parser, arg_parser
from src.actor import EpsilonGreedy
from src.critic import QTable, QDict
from src.experiment import Experiment
from src.wrappers import env_wrappers


if __name__ == "__main__":
    args = arg_parser()
    configs = config_parser(args.config)

    wandb.init(
        project='QL demo',
        mode=args.wandb_mode,
        config=configs,
    )

    env = gym.make(**configs["environment"])
    if 'MiniGrid' in configs["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        critic = QDict(env.observation_space, env.action_space, **configs["critic"])
    else:
        critic = QTable(env.observation_space, env.action_space, **configs["critic"])
    actor = EpsilonGreedy(critic, **configs["actor"])
    experiment = Experiment(env, actor, critic, **configs["experiment"])

    experiment.train()
