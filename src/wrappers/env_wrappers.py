import gymnasium
from gymnasium import spaces
from minigrid import wrappers as minigrid_wrappers
from gymnasium.core import Wrapper


def wrap_minigrid(env):
    env = minigrid_wrappers.FullyObsWrapper(env)
    env = minigrid_wrappers.ImgObsWrapper(env)
    env = NoEdgesWrapper(env)
    env = gymnasium.wrappers.FlattenObservation(env)
    if 'Lava' in env.unwrapped.spec.id:
        env = LavaNoDeath(env)
    env = minigrid_wrappers.ReseedWrapper(env, seeds=(0,))
    return env


class LavaNoDeath(Wrapper):
    """
    Lava cells do not kill the agent. Instead, they yield a penalty of -10.

    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None and current_cell.type == "lava":
            terminated = False
            reward = -10

        return obs, reward, terminated, truncated, info


class NoEdgesWrapper(gymnasium.ObservationWrapper):
    """
    MiniGrid grids are surrounded by empty cells.
    This wrapper removes these cells.

    Args:
        env (gymnasium.Env): the MiniGrid environment.
    """

    def __init__(self, env):
        gymnasium.ObservationWrapper.__init__(self, env)

        obs_space = env.observation_space
        assert len(obs_space.shape) == 3, \
            f'observations must be images (received shape {obs_space.shape})'

        self.observation_space = gymnasium.spaces.Box(
            low=obs_space.low[1:-1, 1:-1],
            high=obs_space.high[1:-1, 1:-1],
            shape=(obs_space.shape[0] - 2, obs_space.shape[1] - 2, obs_space.shape[2]),
            dtype=obs_space.dtype,
        )

    def observation(self, observation):
        return observation[1:-1, 1:-1]
