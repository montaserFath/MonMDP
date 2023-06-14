import gymnasium as gym
from common import utils
from decision_making import get_agent, get_strategy, Experiment

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

    

    # TODO if os.path.exists("api_key.wandb"):
    #     with open("api_key.wandb", 'r') as f:
    #         os.environ["WANDB_API_KEY"] = f.read()
    #         if not configs["online_wandb"]:
    #             os.environ["WANDB_MODE"] = "offline"
