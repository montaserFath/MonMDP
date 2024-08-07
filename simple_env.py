"""Train/evaluate/Plot in Simple/Fire/Button environments"""
import gymnasium as gym
import os
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from src.utils import dict_to_id
from src.wrappers.env_wrappers import TimeStepReward, TabularObservationsWrapper, WindowViewObs, StochasticAction, ChannelsObs
from src.wrappers.monitor_wrappers import BinaryMonitor, StateMonitor, FullMonitor
from src.actor import MonEpsilonGreedyOneAction, MonStateEpsilonGreedy
from src.critic import MonQTableOneAction, StateMonTable, MonQCNN
from src.experiment import MonExperiment
from src.policy_analysis import (
    plot_policy_trajectory,
    plot_reward_table_heatmap,
    plot_q_table_heatmap,
    plot_mdp_mon_q_table_heatmap,
    plot_policy,
    plot_train_joint_reward_timesteps,
)

BASELINES = ["q_learning", "reward_model", "q_monitor_joint", "q_monitor_sequential", "q_mdp", "zero_reward_0"]
ZERO_BASELINES = ["zero_reward_neg", "zero_reward_0", "zero_reward_pos"]
EVAL = False
LOG_DIR = "models/9_9/Penalty/reward_model/env_0.0/eps_1.0/q_lr_1.0/reward_lr_1.0/"
GENERALIZATION = True


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run_monitor(cfg: DictConfig) -> None:
    """Run Monitor Baseline on an env to train or evaluate"""
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
    env = wrappe_env(env_id, train=not EVAL, generalization=GENERALIZATION, monitor_wrapper=True, cfg=cfg)
    env_random, strategy = cfg.environment.random_action_prob, cfg.agent.critic.strategy
    env_size = "3_3" if env_id.split("/")[1].split("-")[-1] == "v0" else "9_9"  # TODO change this
    grid_size = (3, 3) if env_id.split("/")[1].split("-")[-1] == "v0" else (9, 9)  # TODO change this
    train_dir = "general_models/" + env_size + "/" + env_id.split("/")[1].split("-")[1] + "/" + str(strategy) + "/"
    # for hyper-parameters tuning
    if cfg.agent.actor.init_eps != cfg.agent.actor.min_eps:
        raise ValueError("eps should be fixed")
    q_lr, reward_lr, eps = cfg.agent.critic.lr, cfg.agent.critic.reward_model.lr, cfg.agent.actor.init_eps
    train_dir += "env_{}/eps_{}/q_lr_{}/reward_lr_{}/".format(env_random, eps, q_lr, reward_lr)
    os.makedirs(train_dir, exist_ok=True)

    if env_id.split("/")[1].split("-")[2] == "Button":
        if GENERALIZATION:
            critic = MonQCNN(env_id, env.observation_space, env.action_space, dir_name=train_dir, **cfg.agent.critic)
        else:
            critic = StateMonTable(env_id, env.observation_space, env.action_space, train_dir, **cfg.agent.critic)
        actor = MonStateEpsilonGreedy(critic, train=not EVAL, **cfg.agent.actor)
    else:
        if GENERALIZATION:
            critic = MonQCNN(env_id, env.observation_space, env.action_space, dir_name=train_dir, **cfg.agent.critic)
        else:
            critic = MonQTableOneAction(env_id, env.observation_space, env.action_space, train_dir, **cfg.agent.critic)
        actor = MonEpsilonGreedyOneAction(critic, train=not EVAL, **cfg.agent.actor)

    experiment = MonExperiment(env, actor, critic, log_dir=LOG_DIR if EVAL else train_dir, replay_buffer=GENERALIZATION, **cfg.experiment)
    if EVAL:
        for seed in [1]:
            critic.load(LOG_DIR, seed=seed)
            _, _, _, _, _, _, _ = experiment.test(render=False, seed=seed, save_results=True)
            if LOG_DIR.split("/")[3] in ["q_monitor_sequential", "q_monitor_joint"]:
                plot_mdp_mon_q_table_heatmap(log_dir=LOG_DIR, save_fig=True)
            else:
                plot_q_table_heatmap(log_dir=LOG_DIR, grid_size=grid_size, seed=seed, save_fig=True)
            if "reward_model" in LOG_DIR:
                plot_reward_table_heatmap(log_dir=LOG_DIR, save_fig=True)
            plot_policy_trajectory(log_dir=LOG_DIR, env_name=env_id.split("/")[1], seed=seed, save_fig=True)
    else:
        experiment.train()


def wrappe_env(
        env_id: str, train: bool, generalization: bool = False, monitor_wrapper: bool = False, cfg: DictConfig = None,
):
    """Wrapper Simple/Fire/Button env in Monitor MDP or MDP"""
    env = gym.make(env_id, render_modes="human")
    # env = WindowViewObs(env, window_size=(3, 3), grid_size=(10, 10), image_obs=False)
    grid_size = (3, 3) if env_id.split("/")[1].split("-")[-1] == "v0" else (9, 9)  # TODO change this
    if generalization:
        env = ChannelsObs(env, grid_size, n_objects=4 if env_id.split("/")[1].split("-")[2] == "Button" else 3)
    else:
        env = TabularObservationsWrapper(env, grid_size=grid_size)
    env = TimeStepReward(env, timestep_penalty=0.0, goal_reward=1, fire_reward=-10)
    env_random = cfg.environment.random_action_prob if not EVAL else float(LOG_DIR.split("/")[4].split("_")[-1])
    env = StochasticAction(env, random_prob=env_random)
    if monitor_wrapper:
        if train:
            full_monitor = cfg.agent.critic.strategy == "q_learning"
        else:
            full_monitor = LOG_DIR.split("/")[3] == "q_learning"
        if env_id.split("/")[1].split("-")[1] == "Button":
            if cfg.monitor.id != "StateMonitor":
                raise ValueError("For Button env the Monitor should be StateMonitor")
            env = StateMonitor(env, full_monitor, **cfg.monitor)
        else:
            env = BinaryMonitor(env, full_monitor, **cfg.monitor)
    return env


if __name__ == "__main__":
    # plot_train_joint_reward_timesteps("Penalty", ["reward_model"], plot_mean=True, save_fig=True)
    # plot_train_joint_reward_timesteps("Button", BASELINES[:], plot_mean=True, save_fig=True)
    # plot_policy(BASELINES, "Penalty", save_fig=True)
    run_monitor()
