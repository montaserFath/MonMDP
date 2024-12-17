from typing import Optional

import gymnasium as gym
import random
import numpy as np
import torch
import pygame


class PlantsWateringEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
        "torch": True,
        "jax": True,
    }
    def __init__(
        self,
        grid_size: (int, int),
        n_plants: int,
        plants_dryness_prob: float,
        dry_difference: float = 0.5,
        agent_start_pos: [int, int] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        add_new_plants: bool = False,
        add_more_plants: bool = False,
        **kwargs,
    ):

        self.n_plants = n_plants
        self.plants_dryness_prob = plants_dryness_prob
        self.dry_difference = dry_difference
        self.dryness_levels = np.arange(0, 1.1, dry_difference)
        self.agent_start_pos = agent_start_pos
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.add_new_plants = add_new_plants
        self.add_more_plants = add_more_plants
        if self.add_more_plants:
            self._more_plants_values = [i / 8 for i in [1, 2, 3, 5, 6, 7]]

        self._n_raws, self._n_columns = grid_size
        self._n_actions = 6  # 0 up, 1 down, 2 right, 3 left, 4 water, 5 do nothing
        self._n_objects = 3  # 0 agent, 1 plant, 2 dryness, 3 walls
        self._obs_shape = (self._n_objects, self._n_raws, self._n_columns)
        self._grid = np.zeros(self._obs_shape)
        self._agent_pos = None
        self._plants_pos = None
        self._new_plants_pos = None
        self._more_plants_pos = None
        self._n_new_plants = 4 if self.add_new_plants else 0  # TODO: fix this
        if self.add_more_plants:
            self._n_more_plants = 4
            self._n_new_plants = 8
        else:
            self._n_more_plants = 0
        self._previous_agent_pos = None

        self.action_space = gym.spaces.Discrete(self._n_actions)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=self._obs_shape, dtype=np.float64)
        self.reward_range = (-1, 1)

        self._obj_codes = {0: "agent", 1: "plant", 2: "dryness", 3: "wall"}
        self._actions_code = {0: "up", 1: "down", 2: "right", 3: "left", 4: "water", 5: "nothing"}
        self._current_timestep = 0
        self._combinations = np.squeeze(np.meshgrid(range(self._n_raws), range(self._n_columns))).T.reshape(-1, 2)

        # for rendering
        self._window_size = (min(64 * self._n_raws, 512), min(64 * self._n_columns, 512))
        self._cell_size = (self._window_size[0] // self._n_raws, self._window_size[1] // self._n_columns)
        self._clock = pygame.time.Clock()
        self._window_surface = None
        self._colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "brown": (150, 75, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
        }

    def transit(self, obs, action):
        """one-step transition to the env"""
        next_obs = self._grid.copy()
        agent_pos = self._agent_pos.copy()
        if action in np.arange(4):
            next_obs[0, agent_pos[0], agent_pos[1]] = 0  # reset
            agent_pos = self.move(agent_pos, action)
            next_obs[0, agent_pos[0], agent_pos[1]] = 1  # move the agent
        elif action == 4:  # Water
            water_plant = np.any(np.all(agent_pos == self._plants_pos, axis=1))  # Water a plant
            water_new_plant = (
                np.any(np.all(agent_pos == self._new_plants_pos, axis=1))
                if (self.add_new_plants or self.add_more_plants)
                else False
            )  # Water a new plant
            water_diff_plant = (
                np.any(np.all(agent_pos == self._more_plants_pos, axis=1)) if self.add_more_plants else False
            )  # Water a new plant
            if water_plant or water_new_plant or water_diff_plant:
                if next_obs[2, agent_pos[0], agent_pos[1]] > np.min(self.dryness_levels):
                    next_obs[2, agent_pos[0], agent_pos[1]] -= self.dry_difference
        return next_obs

    def step(self, action: int):
        """step function"""
        self._previous_agent_pos = self._agent_pos
        reward = self.reward(action)
        if action in np.arange(4):
            self._grid[0, self._agent_pos[0], self._agent_pos[1]] = 0  # reset
            self._agent_pos = self.move(self._agent_pos, action)
            self._grid[0, self._agent_pos[0], self._agent_pos[1]] = 1  # move the agent
        elif action == 4:  # Water
            water_plant = np.any(np.all(self._agent_pos == self._plants_pos, axis=1))  # Water a plant
            water_new_plant = (
                np.any(np.all(self._agent_pos == self._new_plants_pos, axis=1))
                if (self.add_new_plants or self.add_more_plants)
                else False
            )  # Water a new plant
            water_diff_plant = (
                np.any(np.all(self._agent_pos == self._more_plants_pos, axis=1)) if self.add_more_plants else False
            )  # Water a new plant
            if water_plant or water_new_plant or water_diff_plant:
                if self._grid[2, self._agent_pos[0], self._agent_pos[1]] > np.min(self.dryness_levels):
                    self.water_plant(self._agent_pos)

        # randomly select a plant and increase the dryness level
        if np.random.random() < self.plants_dryness_prob:
            self.update_plants_dryness()
        self._current_timestep += 1
        return self._grid, reward, self._current_timestep >= self.max_episode_steps, False, self.update_info()

    @staticmethod
    def seed(self, seed: int = 0):
        """Seed the environment, numpy, random and torch"""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self, seed: int = None, **kwargs):
        """Reset the environment"""
        super().reset(seed=seed, **kwargs)
        # self.seed(seed)
        self._grid = np.zeros(self._obs_shape)
        self._current_timestep = 0

        # agent & Plants positions
        self.reset_agent_plants_pos()
        self._grid[0, self._agent_pos[0], self._agent_pos[1]] = 1  # agent position
        # Plants location half & half
        self._grid[1, self._plants_pos[:, 0], self._plants_pos[:, 1]] = 1.0

        # Plants dryness
        self._grid[2, self._plants_pos[:, 0], self._plants_pos[:, 1]] = 1

        if self.add_new_plants:
            self._grid[1, self._new_plants_pos[:, 0], self._new_plants_pos[:, 1]] = 0.5  # Cacti
            self._grid[2, self._new_plants_pos[:, 0], self._new_plants_pos[:, 1]] = 1.0  # Cacti are always dry

        if self.add_more_plants:
            self._grid[1, self._new_plants_pos[:, 0], self._new_plants_pos[:, 1]] = 0.5  # Cacti
            self._grid[2, self._new_plants_pos[:, 0], self._new_plants_pos[:, 1]] = 1.0  # Cacti are always dry

            self._grid[1, self._more_plants_pos[:, 0], self._more_plants_pos[:, 1]] = np.random.choice(
                self._more_plants_values,
                self._n_more_plants,
            )
            self._grid[2, self._more_plants_pos[:, 0], self._more_plants_pos[:, 1]] = 1.0  # more Plants are always dry
        self._previous_agent_pos = self._agent_pos
        return self._grid, {
            "agent_pos": self._agent_pos,
            "grid": self._grid,
            "previous_agent_pos": self._previous_agent_pos,
        }

    def render(self):
        return self._render_gui(self.render_mode)

    def close(self):
        """Close the environment"""
        if self._window_surface is not None:
            pygame.display.quit()
            pygame.quit()

    def move(self, agent_pos: list, action: int):
        """Move the agent in the grid"""
        new_agent_pos = [agent_pos[0], agent_pos[1]]
        if action not in np.arange(4):
            raise ValueError("Action must be in range [0, 4)")

        if action == 0:  # Up
            new_agent_pos[0] = max(0, agent_pos[0] - 1)
        elif action == 1:  # Down
            new_agent_pos[0] = min(self._n_raws - 1, agent_pos[0] + 1)
        elif action == 2:  # right
            new_agent_pos[1] = min(self._n_columns - 1, agent_pos[1] + 1)
        else:  # left
            new_agent_pos[1] = max(0, agent_pos[1] - 1)
        return np.array(new_agent_pos)

    def reward(self, action: int) -> float:
        """Get the timestep environment reward"""
        if action == 4:  # Water
            if np.any(np.all(self._agent_pos == self._plants_pos, axis=1)):  # Water a plant
                if self._grid[2, self._agent_pos[0], self._agent_pos[1]] > np.min(self.dryness_levels):
                    return 1.0  # Water a dry plant
                return -1.0  # Water a full watered plan
            if self.add_new_plants or self.add_more_plants:
                tmp_pos = np.concatenate((self._new_plants_pos, self._more_plants_pos), 0)
                pos = self._new_plants_pos if self.add_new_plants else tmp_pos
                if np.any(np.all(self._agent_pos == pos, axis=1)):  # Water a new plant
                    return -1.0
            return -0.2  # Water an empty cell
        return 0.0

    def water_plant(self, plant_pos: list):
        """Water a plant by reducing the amount of dryness by dry_difference"""
        self._grid[2, plant_pos[0], plant_pos[1]] -= self.dry_difference

    def reset_agent_plants_pos(self):
        """Reset the agent and plants position in the grid to the initial position."""
        pos = self._combinations[np.random.choice(self._combinations.shape[0], self.n_plants + 1, replace=False)]
        if self.add_new_plants:
            new_plants = np.squeeze(np.meshgrid(range(self._n_raws), range(3, 6))).T.reshape(-1, 2)
            plants_comb_0 = np.squeeze(np.meshgrid(range(self._n_raws), range(3))).T.reshape(-1, 2)
            plants_comb_1 = np.squeeze(np.meshgrid(range(self._n_raws), range(6, self._n_columns))).T.reshape(-1, 2)
            plants_comb = np.concatenate((plants_comb_0, plants_comb_1), 0)

            self._new_plants_pos = new_plants[np.random.choice(new_plants.shape[0], self._n_new_plants, replace=False)]
            self._plants_pos = plants_comb[np.random.choice(plants_comb.shape[0], self.n_plants, replace=False)]
        elif self.add_more_plants:
            half_plant = self.n_plants // 2 + self._n_new_plants // 2
            monitor_comb = np.squeeze(np.meshgrid(range(self._n_raws), range(self._n_columns // 2))).T.reshape(-1, 2)
            tmp_indx = range(self._n_columns // 2, self._n_columns)
            un_monitor_comb = np.squeeze(np.meshgrid(range(self._n_raws), tmp_indx)).T.reshape(-1, 2)

            mon_pos = monitor_comb[np.random.choice(monitor_comb.shape[0], half_plant, replace=False)]
            un_mon_pos = un_monitor_comb[
                np.random.choice(
                    un_monitor_comb.shape[0],
                    half_plant + self._n_more_plants,
                    replace=False,
                )
            ]

            plant_indx = self.n_plants // 2
            self._plants_pos = np.concatenate((mon_pos[:plant_indx], un_mon_pos[:plant_indx]), 0)
            self._new_plants_pos = np.concatenate(
                (mon_pos[plant_indx:half_plant], un_mon_pos[plant_indx:half_plant]), 0
            )

            self._more_plants_pos = un_mon_pos[half_plant:]

        else:
            self._plants_pos = pos[1:]
        # Agent starting position
        self._agent_pos = [pos[0, 0], pos[0, 1]] if self.agent_start_pos is None else self.agent_start_pos

    def update_plants_dryness(self) -> None:
        """Update plants dryness level"""
        # select a single plant and stochastic
        plant_id = np.random.randint(self.n_plants)
        new_dry = self._grid[2, self._plants_pos[plant_id, 0], self._plants_pos[plant_id, 1]] + self.dry_difference
        self._grid[2, self._plants_pos[plant_id, 0], self._plants_pos[plant_id, 1]] = min(
            np.max(self.dryness_levels),
            new_dry,
        )

    def update_info(self) -> dict:
        """Get information about the environment"""
        info = {}
        info["agent_pos"] = self.get_agent_pos()
        info["grid"] = self.get_grid()
        info["plants_pos"] = self.get_plants_pos()
        info["new_plants_pos"] = self.get_new_plants_pos()
        info["more_plants_pos"] = self.get_more_plants_pos()
        info["previous_agent_pos"] = self._previous_agent_pos
        return info

    def get_grid(self):
        """Get the current grid"""
        return self._grid.copy()

    def get_new_plants_pos(self):
        """Get cactus position inside the grid"""
        return self._new_plants_pos

    def get_more_plants_pos(self):
        """Get cactus position inside the grid"""
        return self._more_plants_pos

    def get_agent_pos(self):
        """Get the current agent position"""
        return self._agent_pos

    def get_plants_pos(self):
        """Get plants position inside the grid"""
        return self._plants_pos

    def _draw_lines(self):
        """Draw lines between cells"""
        # white lines between cells
        cell_x, cell_y = self._cell_size
        # horizontal lines
        for i in range(1, self._n_raws):
            pygame.draw.line(self._window_surface, self._colors["black"], (0, i * cell_x), (cell_x**2, i * cell_y), 3)
        # vertical lines
        for j in range(1, self._n_columns):
            pygame.draw.line(self._window_surface, self._colors["black"], (j * cell_y, 0), (j * cell_x, cell_y**2), 3)

    def _render_gui(self, mode):
        """render gui"""
        if self._window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Plants Watering Environment")
                self._window_surface = pygame.display.set_mode(self._window_size)
            elif mode == "rgb_array":
                self._window_surface = pygame.Surface(self._window_size)
            else:
                raise ValueError("Undefined render mode")
        self._window_surface.fill(self._colors["white"])
        self._draw_lines()
        for obj_id in range(self.n_plants + 1):
            if obj_id == self.n_plants:
                self._draw_obj("agent", self._agent_pos)
            else:
                plant_pos = self._plants_pos[obj_id]
                plant_type = "plant_0" if self._grid[1, plant_pos[0], plant_pos[1]] == 1.0 else "plant_1"
                self._draw_obj(plant_type, plant_pos, self._grid[2, plant_pos[0], plant_pos[1]])
        if self.add_new_plants:
            for new_plant in range(self._n_new_plants):
                new_plants_pos = self._new_plants_pos[new_plant]
                self._draw_obj("new_plant", new_plants_pos, self._grid[2, new_plants_pos[0], new_plants_pos[1]])

        if self.add_more_plants:
            for new_plant in range(self._n_new_plants):
                new_plants_pos = self._new_plants_pos[new_plant]
                self._draw_obj("new_plant", new_plants_pos, self._grid[2, new_plants_pos[0], new_plants_pos[1]])

            for more_plant in range(self._n_more_plants):
                more_plants_pos = self._more_plants_pos[more_plant]
                self._draw_obj("more_plant", more_plants_pos, self._grid[2, more_plants_pos[0], more_plants_pos[1]])

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self._window_surface)), axes=(1, 0, 2))
        else:
            raise NotImplementedError

    def _draw_obj(self, obj: str, pos: [int, int], dryness: float = 0.0):
        """Draw the grid"""
        # TODO solve flipping x and y
        new_pos = ((pos[1] + 0.5) * self._cell_size[0], (pos[0] + 0.5) * self._cell_size[1])
        shift_x, shift_y = 0.5 * self._cell_size[0], 0.5 * self._cell_size[1]
        if obj in ["plant_0", "plant_1"]:
            # circle or square plants
            if obj == "plant_0":
                pygame.draw.circle(self._window_surface, self._get_object_color(dryness), new_pos, 20)
            else:
                pygame.draw.rect(
                    self._window_surface,
                    self._get_object_color(dryness),
                    (new_pos[0] - shift_x, new_pos[1] - shift_y, 50, 50),
                    0,
                )
        elif obj == "agent":
            pygame.draw.rect(self._window_surface, self._colors["blue"], (new_pos[0], new_pos[1], 30, 30), 0)
        elif obj == "more_plant":
            loc = [
                [new_pos[0] + 20, new_pos[1] - 20],
                [new_pos[0] - 20, new_pos[1]],
                [new_pos[0] + 20, new_pos[1] + 20],
            ]
            pygame.draw.polygon(self._window_surface, self._get_object_color(dryness), loc)
        elif obj == "new_plant":
            loc = [
                [new_pos[0] - 20, new_pos[1] + 20],
                [new_pos[0], new_pos[1] - 20],
                [new_pos[0] + 20, new_pos[1] + 20],
            ]
            pygame.draw.polygon(self._window_surface, self._get_object_color(dryness), loc)
        else:
            raise ValueError("Undefined object type")

    def _get_object_color(self, dryness: float) -> str:
        """Get plant color red, brown or green"""
        if dryness == np.max(self.dryness_levels):
            return self._colors["red"]
        if dryness == np.min(self.dryness_levels):
            return self._colors["green"]
        return self._colors["brown"]
