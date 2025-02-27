"""gym register for MDP and Monitored MDP environments"""
from gymnasium.envs.registration import register


def register_envs():
    """gym register for MDP and Monitored MDP environments"""

    register(
        id="Plants-Watering-v1",
        entry_point="gym_monitor.plant_watering_env:PlantsWateringEnv",
        max_episode_steps=100,
    )
