from gymnasium.envs.registration import register

register(
    id='custom_envs/ToyWorld-v0',
    entry_point='custom_envs.envs:ToyWorld',
)