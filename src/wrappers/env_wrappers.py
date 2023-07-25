import gymnasium
from minigrid import wrappers as minigrid_wrappers


def wrap_minigrid(env):
    env = minigrid_wrappers.FullyObsWrapper(env)
    env = minigrid_wrappers.ImgObsWrapper(env)
    env = gymnasium.wrappers.FlattenObservation(env)
    if 'Lava' in env.unwrapped.spec.id:
        env = minigrid_wrappers.NoDeath(env)
    env = minigrid_wrappers.ReseedWrapper(env, seeds=(0,))
    return env
