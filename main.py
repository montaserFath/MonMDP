import gymnasium
from gymnasium.spaces import Discrete
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import dict_to_id
from src.actor import MonEpsilonGreedy
from src.critic import MonQDict, MonQTable
from src.experiment import MonExperiment
from src.wrappers import env_wrappers, monitor_wrappers


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg : DictConfig) -> None:
    group = cfg["environment"]["id"] + '\\' + dict_to_id(cfg["monitor"])
    wandb.init(
        group=group,
        config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True,
        ),
        settings=wandb.Settings(
            start_method="thread"
        ),
        **cfg["wandb"],
    )

    env = gymnasium.make(**cfg["environment"])
    if 'MiniGrid' in cfg["environment"]["id"]:
        env = env_wrappers.wrap_minigrid(env)
    env = getattr(monitor_wrappers, cfg["monitor"]["id"])(env, **cfg["monitor"])

    if isinstance(env.env.observation_space, Discrete):
        critic = MonQTable(env.observation_space, env.action_space, **cfg["agent"]["critic"])
    else:
        critic = MonQDict(env.observation_space, env.action_space, **cfg["agent"]["critic"])

    actor = MonEpsilonGreedy(critic, **cfg["agent"]["actor"])

    experiment = MonExperiment(env, actor, critic, **cfg["experiment"])

    experiment.train()


if __name__ == "__main__":
    run()
