import gymnasium as gym
import wandb

from src.utils import config_parser, arg_parser
from src.actor import MonEpsilonGreedy
from src.critic import MonQDict, MonQTable
from src.experiment import MonExperiment
from src.wrappers import env_wrappers, monitor_wrappers


if __name__ == "__main__":
    args = arg_parser()
    configs = config_parser(args.config)

    wandb.init(
        entity="ualberta-bowling",
        project="QL demo",
        mode=args.wandb_mode,
        config=configs,
    )

    env = gym.make(**configs["environment"])
    if 'MiniGrid' in configs["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        env = monitor_wrappers.BinaryMonitor(env)
        critic = MonQDict(env.observation_space, env.action_space, **configs["critic"])
    else:
        env = monitor_wrappers.BinaryMonitor(env)
        critic = MonQTable(env.observation_space, env.action_space, **configs["critic"])
    actor = MonEpsilonGreedy(critic, **configs["actor"])
    experiment = MonExperiment(env, actor, critic, **configs["experiment"])

    experiment.train()
