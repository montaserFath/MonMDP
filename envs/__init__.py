from gymnasium.envs.registration import register

register(
    id='monitor/ToyWorld-v0',
    entry_point='envs.custom:ToyWorld',
)

register(
    id='monitor/MonitorGrid-v0',
    entry_point='envs.custom:MonitorGrid',
)

register(
    id='monitor/LavaCrossingNoDeathS5N1-v0',
    entry_point='envs.custom:LavaCrossingNoDeath',
    kwargs={"size": 5, "num_crossings": 1},
)
