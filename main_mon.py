import gymnasium as gym

from src.utils import config_parser, arg_parser, init_wandb
from src.actor import MonEpsilonGreedy
from src.critic import MonQDict
from src.experiment import MonExperiment
from src.wrappers import env_wrappers, monitor_wrappers


if __name__ == "__main__":
    args = arg_parser()
    configs = config_parser(args.config)
    init_wandb(args.wandb_mode)

    env = gym.make(**configs["environment"])
    if 'MiniGrid' in configs["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
        env = monitor_wrappers.RandomMonitor(env)
        critic = MonQDict(env.observation_space, env.action_space, **configs["critic"])
    else:
        env = monitor_wrappers.RandomMonitor(env)
        raise NotImplementedError
    actor = MonEpsilonGreedy(critic, **configs["actor"])
    experiment = MonExperiment(env, actor, critic, **configs["experiment"])

    experiment.train()
