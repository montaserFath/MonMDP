import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from src.wrappers.env_wrappers import (
    TimeStepReward,
    TabularObservationsWrapper,
    StableBaselinesWrapper,
)


def main():
    """main function"""
    env = gym.make("gym_monitor/TreasureHunt-Fire-v0", render_modes="human")
    env = TabularObservationsWrapper(env, grid_size=(3, 3))
    env = TimeStepReward(env, decay_rate=0.01)
    env = StableBaselinesWrapper(env)
    # model = train_dqn(env, log_dir="models/simple_env/something")
    evaluate_mode("models/simple_env/dqn_fire_test.zip", env)

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


def evaluate_mode(model_dir: str, env, n_episodes: int = 10):
    """Evaluate a DQN model in an environment"""
    model = DQN.load(model_dir)
    for ep in range(n_episodes):
        done = False
        total_r, t = 0, 0
        s, _ = env.reset()
        img = env.render()
        # np.save("models/simple_env/simple_env_img.npy", img)
        # print("init state", s[0])
        while not done:
            # a = env.action_space.sample()
            a, _ = model.predict(s, deterministic=True)
            s, r, done, _, _ = env.step(a)
            total_r += r
            t += 1
        print("Episode: {}, Reward: {}, timesteps: {}".format(ep, total_r, t))


if __name__ == "__main__":
    main()
