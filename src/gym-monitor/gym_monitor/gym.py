from gymnasium.envs.registration import register

def register_envs():
    register(
        id="ToyChain-v0",
        entry_point="gym_monitor.chain:ToyChain",
        max_episode_steps=100,
    )
    register(
        id="ToyGrid-4x8-v0",
        entry_point="gym_monitor.grid:ToyGrid",
        max_episode_steps=100,
        kwargs={"map": "4x8"},
    )
