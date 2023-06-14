import gymnasium as gym
from common import utils
from decision_making import get_agent, get_strategy, Experiment
import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    prompt_params = utils.get_prompts()
    configs = utils.get_configs(prompt_params.configs_name)
    env = gym.make(configs["environment"])
    if isinstance(env.observation_space, gym.spaces.Discrete):
        configs["agent"].update({"n_states": env.observation_space.n})
    else:
        raise NotImplementedError
    configs["agent"].update({"n_actions": env.action_space.n})
    print("params:", configs)

    agent = get_agent(**configs["agent"])
    strategy = get_strategy(agent=agent, **configs["strategy"])
    experiment = Experiment(env, agent, strategy, **configs["experiment"])
    logs = experiment.run()

    logs = np.asarray(logs)
    mean_return = np.mean(logs, axis=0)
    std_return = np.std(logs, axis=0)
    lower_bound = mean_return - 2 * std_return / math.sqrt(configs["experiment"]["n_runs"])
    upper_bound = mean_return + 2 * std_return / math.sqrt(configs["experiment"]["n_runs"])

    plt.fill_between(np.arange(float(configs["experiment"]["n_episodes"])), lower_bound, upper_bound, alpha=0.25)
    plt.plot(np.arange(float(configs["experiment"]["n_episodes"])), mean_return, alpha=1, color="k")
    plt.xlabel("episodes")
    plt.ylabel("return")
    plt.title(f"online performance over {configs['experiment']['n_runs']} runs")
    plt.grid()
    plt.show()


    # TODO if os.path.exists("api_key.wandb"):
    #     with open("api_key.wandb", 'r') as f:
    #         os.environ["WANDB_API_KEY"] = f.read()
    #         if not configs["online_wandb"]:
    #             os.environ["WANDB_MODE"] = "offline"
