import gymnasium as gym
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from src.utils import dict_to_id
from src.wrappers.env_wrappers import TimeStepReward, TabularObservationsWrapper
from src.wrappers.monitor_wrappers import BinaryMonitor, StateMonitor
from src.actor import MonEpsilonGreedyOneAction, MonStateEpsilonGreedy
from src.critic import MonQTableOneAction, StateMonTable
from src.experiment import MonExperiment
from src.policy_analysis import (
    plot_policy_trajectory,
    plot_reward_table_heatmap,
    plot_q_table_heatmap,
    plot_mdp_mon_q_table_heatmap,
    plot_policy,
    plot_train_joint_reward_timesteps,
)


EVAL = True
LOG_DIR = "models/Switch/reward_model/"


@hydra.main(version_base=None, config_path="configs", config_name="switch_env")
def run_monitor(cfg: DictConfig) -> None:
    group = cfg.environment.id + "\\" + dict_to_id(cfg.monitor)
    if not EVAL:
        wandb.init(
            group=group,
            config=OmegaConf.to_container(
                cfg,
                resolve=True,
                throw_on_missing=True,
            ),
            settings=wandb.Settings(start_method="thread"),
            **cfg.wandb,
        )
    env_id = cfg["environment"]["id"]
    env = wrappe_env(env_id, train=not EVAL, monitor_wrapper=True, cfg=cfg)

    if env_id.split("/")[1].split("-")[1] == "Switch":
        critic = StateMonTable(env_id, env.observation_space, env.action_space, **cfg.agent.critic)
        actor = MonStateEpsilonGreedy(critic, train=not EVAL, **cfg.agent.actor)
    else:
        critic = MonQTableOneAction(env_id, env.observation_space, env.action_space, **cfg.agent.critic)
        actor = MonEpsilonGreedyOneAction(critic, train=not EVAL, **cfg.agent.actor)
    train_dir = "models/" + env_id.split("/")[1].split("-")[1] + "/" + str(cfg.agent.critic.strategy) + "/"
    experiment = MonExperiment(env, actor, critic, log_dir=train_dir, **cfg.experiment)
    if EVAL:
        critic.load(LOG_DIR, seed=cfg.experiment.rng_seed)
        _, _, _, _, _, _ = experiment.test(render=False, save_results=True)
        if LOG_DIR.split("/")[-2] in ["q_monitor_sequential", "q_monitor_joint"]:
            plot_mdp_mon_q_table_heatmap(log_dir=LOG_DIR, save_fig=True)
        else:
            plot_q_table_heatmap(log_dir=LOG_DIR, save_fig=True)
        if "reward_model" in LOG_DIR:
            plot_reward_table_heatmap(log_dir=LOG_DIR, save_fig=True)
        plot_policy_trajectory(log_dir=LOG_DIR, env_name=env_id.split("/")[1], save_fig=True)
    else:
        experiment.train()


def wrappe_env(env_id: str, train: bool, monitor_wrapper: bool = False, cfg: DictConfig = None):
    env = gym.make(env_id, render_modes="human")
    env = TabularObservationsWrapper(env, grid_size=(3, 3))
    env = TimeStepReward(env, timestep_penalty=0.0, goal_reward=1, fire_reward=-10)
    if monitor_wrapper:
        if train:
            full_monitor = cfg.agent.critic.strategy == "q_learning"
        else:
            full_monitor = LOG_DIR.split("/")[-2] == "q_learning"
        if env_id.split("/")[1].split("-")[1] == "Switch":
            if cfg.monitor.id != "StateMonitor":
                raise ValueError("For Switch env the Monitor should be StateMonitor")
            env = StateMonitor(env, full_monitor, **cfg.monitor)
        else:
            env = BinaryMonitor(env, full_monitor, **cfg.monitor)
    return env


if __name__ == "__main__":
    baselines = ["q_learning", "reward_model", "q_monitor_joint", "q_monitor_sequential", "q_mdp", "zero_reward"]
    # plot_train_joint_reward_timesteps("Switch", baselines[:], plot_mean=True, save_fig=True)
    # plot_policy(baselines[1:], "Switch", save_fig=True)
    run_monitor()
