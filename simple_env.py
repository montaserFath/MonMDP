import gymnasium as gym
from src.wrappers.env_wrappers import TimeStepReward, TabularObservationsWrapper


def main():
    """main function"""
    env = gym.make("gym_monitor/TreasureHunt-Simple-v0", render_modes="human")
    env = TabularObservationsWrapper(env, grid_size=(3, 3))
    env = TimeStepReward(env, decay_rate=0.01)
    episodes = 5
    for ep in range(episodes):
        done = False
        total_r, t = 0, 0
        s = env.reset()
        env.render()
        # print("init state", s[0])
        while not done:
            a = env.action_space.sample()
            s, r, done, _, _ = env.step(a)
            total_r += r
            t += 1
        print("Episode: {}, Reward: {}, timesteps: {}".format(ep, total_r, t))
    env.close()


if __name__ == "__main__":
    main()
