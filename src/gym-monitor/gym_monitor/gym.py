from gymnasium.envs.registration import register

def register_envs():
    register(
        id="ToyChain-v0",
        entry_point="gym_monitor.chain:ToyChain",
    )
    register(
        id="ToyGrid-v0",
        entry_point="gym_monitor.grid:ToyGrid",
    )
