from gymnasium.envs.registration import register

register(
    id='monitor/ToyWorld-v0',
    entry_point='envs.custom:ToyWorld',
)

register(
    id='monitor/MonitorGrid-v0',
    entry_point='envs.custom:MonitorGrid',
)
