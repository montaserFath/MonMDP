from .agents import *
from .action_selection_strategies import *
from .experiments import TrainExperiment, TestExperiment

AGENTS = dict(QLearningAgent=QLearningAgent,
              )
AS_STRATEGIES = dict(EpsilonGreedy=EpsilonGreedy,
                  )


def get_agent(**kwargs) -> BaseAgent:
    return AGENTS[kwargs["agent_name"]](**kwargs)


def get_as_strategy(**kwargs) -> BaseStrategy:
    return AS_STRATEGIES[kwargs["as_strategy_name"]](**kwargs)
