from gymnasium.envs.registration import register

def register_envs():

    register(
        id="TreasureHunt-Easy-v0",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=100,
        kwargs={
            "grid": "4x8",
            "enable_map": False,
            "enable_quicksand": False
            },
    )

    register(
        id="TreasureHunt-Medium-v0",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=200,
        kwargs={
            "grid": "4x8",
            "enable_map": False,
            "enable_quicksand": True
            },
    )

    register(
        id="TreasureHunt-Hard-v0",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=500,
        kwargs={
            "grid": "4x8",
            "enable_map": True,
            "enable_quicksand": True
            },
    )
