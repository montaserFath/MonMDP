"""gym register for MDP and Monitored MDP environments"""
from gymnasium.envs.registration import register


def register_envs():
    """gym register for MDP and Monitored MDP environments"""
    register(
        id="TreasureHunt-Simple-v0",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3",
            "init_agent_pos": (0, 0),
            "render_mode": "human",
        },
    )

    register(
        id="TreasureHunt-Penalty-v0",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3 penalty",
            "init_agent_pos": (0, 0),
            "render_mode": "human",
        },
    )

    register(
        id="TreasureHunt-Button-v0",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3 button",
            "init_agent_pos": (0, 0),
            "render_mode": "human",
        },
    )

    register(
        id="TreasureHunt-Penalty-v1",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=5000,
        kwargs={
            "grid": "9x9 penalty",
            "init_agent_pos": (0, 0),
            "render_mode": "human",
        },
    )

    register(
        id="TreasureHunt-Button-v1",
        entry_point="gym_monitor.treasure_hunt:TreasureHunt",
        max_episode_steps=5000,
        kwargs={
            "grid": "9x9 button",
            "init_agent_pos": (0, 0),
            "render_mode": "human",
        },
    )

    register(
        id="Plants-Watering-v1",
        entry_point="gym_monitor.plant_watering_env:PlantsWateringEnv",
        max_episode_steps=100,
    )
