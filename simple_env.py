import numpy as np
import gymnasium as gym
import hydra
from stable_baselines3 import DQN
import wandb
from omegaconf import DictConfig, OmegaConf
from src.utils import dict_to_id
from src.wrappers.env_wrappers import (
    TimeStepReward,
    TabularObservationsWrapper,
    StableBaselinesWrapper,
)
from src.wrappers.monitor_wrappers import BinaryMonitor, StateMonitor
from src.actor import MonEpsilonGreedyOneAction, MonStateEpsilonGreedy
from src.critic import MonQTableOneAction, StateMonTable
from src.experiment import MonExperiment
from src.policy_analysis import (
    plot_policy_actions,
    plot_reward_table_heatmap,
    plot_q_table_heatmap,
    plot_joint_reward,
    plot_mdp_mon_q_table_heatmap,
    plot_joint_reward_seeds,
    plot_joint_reward_seeds_baselines,
)


EVAL = False
LOG_DIR = "models/Switch/reward_model/"


@hydra.main(version_base=None, config_path="configs", config_name="default")
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
    env = wrappe_env(env_id, monitor_wrapper=True, cfg=cfg)

    if env_id.split("/")[1].split("-")[1] == "Switch":
        critic = StateMonTable(env_id, env.observation_space, env.action_space, **cfg.agent.critic)
        actor = MonStateEpsilonGreedy(critic, train=not EVAL, **cfg.agent.actor)
    else:
        critic = MonQTableOneAction(env_id, env.observation_space, env.action_space, **cfg.agent.critic)
        actor = MonEpsilonGreedyOneAction(critic, train=not EVAL, **cfg.agent.actor)
    train_dir = "models/" + env_id.split("/")[1].split("-")[1] + "/" + str(cfg.agent.critic.strategy) + "/"
    experiment = MonExperiment(env, actor, critic, log_dir=train_dir, **cfg.experiment)
    if EVAL:
        critic.load(LOG_DIR)
        _, _, _, _, _, _ = experiment.test(render=False, save_results=True)
        if LOG_DIR.split("/")[-2] in ["q_monitor_sequential", "q_monitor_joint"]:
            plot_mdp_mon_q_table_heatmap(log_dir=LOG_DIR, save_fig=True)
        else:
            plot_q_table_heatmap(log_dir=LOG_DIR, save_fig=True)
        if "reward_model" in LOG_DIR:
            plot_reward_table_heatmap(log_dir=LOG_DIR, save_fig=True)
        plot_policy_actions(log_dir=LOG_DIR, env_name=env_id.split("/")[1], save_fig=True)
        # log_results()

    else:
        experiment.train()


def log_results(
    ep_return_true: np.ndarray,
    ep_return_proxy: np.ndarray,
    ep_return_cost: np.ndarray,
    ep_monitor_action: np.ndarray,
    ep_length: np.ndarray,
) -> None:
    print("Mean True reward: {:.3f}".format(np.mean(ep_return_true)))
    print("Mean Proxy reward: {:.3f}".format(np.mean(ep_return_proxy)))
    print("Mean Cost reward: {:.3f}".format(np.mean(ep_return_cost)))
    print("Mean Episode Monitor Action: {:.2f}".format(np.mean(ep_monitor_action)))
    print("Mean Episode length: {:.2f}".format(np.mean(ep_length)))


def wrappe_env(env_id: str, monitor_wrapper: bool = False, cfg: DictConfig = None):
    env = gym.make(env_id, render_modes="human")
    env = TabularObservationsWrapper(env, grid_size=(3, 3))
    env = TimeStepReward(env, timestep_penalty=0.0, goal_reward=1, fire_reward=-10)
    env = StableBaselinesWrapper(env)
    if monitor_wrapper:
        if env_id.split("/")[1].split("-")[1] == "Switch":
            env = StateMonitor(env, **cfg.monitor)
        else:
            env = BinaryMonitor(env, **cfg.monitor)
    return env


def main():
    """main function"""
    env = wrappe_env("gym_monitor/TreasureHunt-Fire-v0", monitor_wrapper=True)
    # model = train_dqn(env, log_dir="models/simple_env/something")
    evaluate_mode("models/simple_env/dqn_fire_test.zip", env, render=True)

    env.close()


def train_dqn(env, log_dir: str):
    """train an env with DQN"""
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_starts=5000,
        tensorboard_log=log_dir,
        exploration_fraction=0.2,
    )
    model.learn(total_timesteps=int(5e5), log_interval=10)
    model.save(log_dir)
    return model


def evaluate_mode(model_dir: str, env, n_episodes: int = 5, render: bool = False):
    """Evaluate a DQN model in an environment"""
    model = DQN.load(model_dir)
    for ep in range(n_episodes):
        done = False
        total_r, t = 0, 0
        s, _ = env.reset()
        while not done:
            if render:
                env.render()
            # a = env.action_space.sample()
            a, _ = model.predict(s, deterministic=True)
            s, r, done, _, _ = env.step(a)
            total_r += r
            t += 1
        print("Episode: {}, Reward: {}, timesteps: {}".format(ep, total_r, t))


if __name__ == "__main__":
    baselines = ["reward_model", "q_monitor_joint", "q_monitor_sequential", "q_mdp", "zero_reward"]
    # plot_joint_reward_seeds_baselines("Fire", baselines, save_fig=True)
    # plot_joint_reward_seeds("Simple", baselines[0], save_fig=True)
    # plot_joint_reward(baselines, env_name="Simple", save_fig=True)
    run_monitor()
    # main()
