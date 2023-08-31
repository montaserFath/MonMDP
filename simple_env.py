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
from src.wrappers.monitor_wrappers import BinaryMonitor
from src.actor import MonEpsilonGreedyOneAction
from src.critic import MonQTableOneAction
from src.experiment import MonExperiment

EVAL = False


@hydra.main(version_base=None, config_path="configs", config_name="simple_env")
def run_monitor(cfg: DictConfig) -> None:
    group = cfg.environment.id + "\\" + dict_to_id(cfg.monitor)
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

    env = wrappe_env(cfg["environment"]["id"], monitor_wrapper=True, cfg=cfg)

    critic = MonQTableOneAction(env.observation_space, env.action_space, **cfg.agent.critic)

    actor = MonEpsilonGreedyOneAction(critic, train=not EVAL, **cfg.agent.actor)

    experiment = MonExperiment(env, actor, critic, log_dir="models/simple_env/", **cfg.experiment)
    if EVAL:
        critic.load("models/simple_env/2023_08_31-15_13_29")
        log_results(experiment.test(render=True))

    else:
        experiment.train()


def log_results(
        ep_return_true: np.ndarray, ep_return_proxy: np.ndarray, ep_return_cost: np.ndarray, ep_monitor_action: np.ndarray, ep_length: np.ndarray
) -> None:
    print("Mean True reward: {:.3f}".format(np.mean(ep_return_true)))
    print("Mean Proxy reward: {:.3f}".format(np.mean(ep_return_proxy)))
    print("Mean Cost reward: {:.3f}".format(np.mean(ep_return_cost)))
    print("Mean Episode Monitor Action: {:.2f}".format(np.mean(ep_monitor_action)))
    print("Mean Episode length: {:.2f}".format(np.mean(ep_length)))


def wrappe_env(
    env_id: str = "gym_monitor/TreasureHunt-Simple-v0", monitor_wrapper: bool = False, cfg: DictConfig = None,
):
    env = gym.make(env_id, render_modes="human")
    env = TabularObservationsWrapper(env, grid_size=(3, 3))
    env = TimeStepReward(env, decay_rate=0.01)
    env = StableBaselinesWrapper(env)
    if monitor_wrapper:
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
    run_monitor()
    # main()
