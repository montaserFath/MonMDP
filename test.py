import gymnasium
from minigrid import wrappers as minigrid_wrappers


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
            low=obs_space.low[1:-1,1:-1],
            high=obs_space.high[1:-1,1:-1],
            shape=(obs_space.shape[0] - 2, obs_space.shape[1] - 2, obs_space.shape[2]),
            dtype=obs_space.dtype,
        )

    def observation(self, observation):
        return observation[1:-1,1:-1]



if __name__ == '__main__':
    env_name = 'MiniGrid-DoorKey-5x5-v0'

    env = gymnasium.make(env_name, render_mode='human') # (default) 7x7x3 partial obs
    # to visualize, pass render_mode='human'



    env = minigrid_wrappers.FullyObsWrapper(env) # WxHx3 full obs, size depends on the grid
    # env = minigrid_wrappers.RGBImgObsWrapper(env) # if we want RGB-like full obs
    env = minigrid_wrappers.ImgObsWrapper(env) # (mandatory) removes the 'mission' field
    env = NoEdgesWrapper(env)
    env = minigrid_wrappers.ReseedWrapper(env, seeds=(0,)) # this way we don't have to manually fix the seed and we'll have the same env at every reset

    obs, info = env.reset()

    # obs are 3x3x3 arrays, because 3x3 is the size of the grid (after removing the empty edges)
    # and then each cell has an ID with:
    # - what it's in the cell (agent, door, key, ...)
    # - the direction of the agent (if the agent is there)
    # - the color of the object (if there is an object)

    # since the env is fixed thanks to the seed, this obs can be uniquely associated with a state
    # and we can have tabular value functions and policies
