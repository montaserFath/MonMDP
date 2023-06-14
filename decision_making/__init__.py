from .agents import *
from .strategies import *
from .experiments import Experiment

AGENTS = dict(QLearningAgent=QLearningAgent,
              )
STRATEGIES = dict(EpsilonGreedy=EpsilonGreedy,
                  )


def get_agent(**kwargs) -> BaseAgent:
    return AGENTS[kwargs["agent_name"]](**kwargs)


def get_strategy(**kwargs) -> BaseStrategy:
    return STRATEGIES[kwargs["strategy_name"]](**kwargs)
