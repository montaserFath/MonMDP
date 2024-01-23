# pylint: disable=no-member
"""Genral Environmnet for MDP and Minotred MDP"""
from contextlib import closing
from typing import Optional
from io import StringIO
import numpy as np
import gymnasium as gym
import pygame


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

EMPTY = 0
AGENT = 1
GLD_COIN = 2
CRSD_COIN = 3
BUTTON = 4  # 7
WALL = 5  # 8

INT_TO_ANSI = {
    EMPTY: b"E",
    AGENT: b"A",
    GLD_COIN: b"G",
    CRSD_COIN: b"C",
    BUTTON: b"B",
    WALL: b"W",
}

GRIDS = {
    "3x3 penalty": [
        [EMPTY, CRSD_COIN, GLD_COIN],
        [EMPTY, CRSD_COIN, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ],
    "3x3 button": [
        [EMPTY, CRSD_COIN, GLD_COIN],
        [EMPTY, CRSD_COIN, EMPTY],
        [EMPTY, EMPTY, BUTTON],
    ],
    "9x9 penalty": [
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, GLD_COIN],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "9x9 button": [
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, GLD_COIN],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, BUTTON],
    ],
}


def _move(row, col, action, nrow, ncol):
    if action == LEFT:
        col = max(col - 1, 0)
    elif action == DOWN:
        row = min(row + 1, nrow - 1)
    elif action == RIGHT:
        col = min(col + 1, ncol - 1)
    elif action == UP:
        row = max(row - 1, 0)
    else:
        raise ValueError("illegal action")
    return (row, col)


# pylint: disable=too-many-instance-attributes
class TreasureHunt(gym.Env):
    """
    Gridworld where the agent has to find golden coins while avoiding cursed coins.

    ## Versions
    - Easy: coins are always visible to the agent.
    - Medium: some cells have quicksand, where the agent gets
      stuck and has only 5% chance of escaping.
    - Hard: in addition to the presence of quicksand, the agent must
      first find a map to reveal the coins.

    ## Grid
    The grid is defined by a 2D array of integers. It is possible to define
    custom grids.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}`.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    If the agent is in a quicksand cell any action will fail with 95% probability.

    ## Observation Space
    The observation shape is `(rows * cols, )` denoting the flattened grid.
    Each element of the observation is in the range `{0, 6}` denoting the
    content of a cell.

    - 0: Empty
    - 1: Agent
    - 2: Golden coin (reward)
    - 3: Cursed coin (penalty)
    - 4: Quicksand
    - 5: Quicksand with agent in it
    - 6: Map

    ## Starting State
    The game starts with the agent at any of the leftmost cells.

    ## Rewards
    - Collect golden coin: +1
    - Collect cursed coin: -1
    - Otherwise: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. All golden coins have been collected.

    - Truncation:
        1. The length of the episode is 100 for the Easy version.
        2. The length of the episode is 200 for the Medium version.
        3. The length of the episode is 500 for the Hard version.

    ## Rendering
    Human mode renders the environment as a grid with colored cells.

    - Black: empty cells
    - Green: golden coins
    - Red: cursed coins
    - Orange: quicksand
    - White: map

    The agent is the blue circle.

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        render_mode: Optional[str] = None,
        grid: Optional[str] = "4x8",
        init_agent_pos: Optional[tuple] = None,
        location_random: Optional[bool] = False,
        **kwargs,
    ):
        self._location_random = location_random
        self._grid_key = grid
        self._grid = np.asarray(GRIDS[self._grid_key])

        self._n_rows, self._n_cols = self._grid.shape
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(self._n_rows * self._n_cols,), dtype=int)
        self.action_space = gym.spaces.Discrete(4)
        self._init_agent_pos = init_agent_pos
        self._agent_pos = None
        self._last_action = None

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (min(64 * self._n_cols, 512), min(64 * self._n_rows, 512))
        self.cell_size = (self.window_size[0] // self._n_cols, self.window_size[1] // self._n_rows)
        self._current_timestep = 0
        self.info = {}

    @property
    def grid(self) -> np.ndarray:
        """get the current grid status"""

        grid = self._grid.copy()
        grid[grid == GLD_COIN] = EMPTY
        grid[grid == CRSD_COIN] = EMPTY

        return grid

    def reset_grid(self) -> None:
        """reset the gird & add current agent position"""
        self._grid = np.asarray(GRIDS[self._grid_key])
        self._grid[self._agent_pos] = AGENT

    def reset(self, seed: int = None, **kwargs):
        """reset the environment"""
        super().reset(seed=seed, **kwargs)
        self._agent_pos = (
            (self.np_random.integers(self._n_rows), 0) if self._init_agent_pos is None else self._init_agent_pos
        )
        self.reset_grid()
        self._last_action = None
        self._current_timestep = 0
        self.info = {}
        return self.grid.flatten(), {}

    def step(self, action: int):
        prev_agent_pos = self._agent_pos
        self._agent_pos = _move(
            self._agent_pos[0],
            self._agent_pos[1],
            action,
            self._n_rows,
            self._n_cols,
        )
        reward = self.reward()
        # do nothing if the agent steps into a wall
        if self._grid[self._agent_pos] == WALL:
            self._agent_pos = prev_agent_pos
        self.reset_grid()

        terminated = (self._grid != GLD_COIN).all()

        self._last_action = action
        self._current_timestep += 1
        self.info["agent_pos"] = self._agent_pos
        return self.grid.flatten(), reward, terminated, False, self.info

    def reward(self) -> float:
        """
        Reward function returns 1 if the agent collects gold coin, -1 if the agent collects cursed coin, otherwise 0
        """
        if self._agent_pos in list(zip(*np.where(np.asarray(GRIDS[self._grid_key]) == GLD_COIN))):
            return 1.0
        if self._agent_pos in list(zip(*np.where(np.asarray(GRIDS[self._grid_key]) == CRSD_COIN))):
            return -1.0
        return 0.0

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "ansi":
            return self._render_text()
        # self.render_mode in {"human", "rgb_array"}:
        return self._render_gui(self.render_mode)

    # pylint: disable=too-many-branches
    def _render_gui(self, mode):
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Treasure Hunt")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None, "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        grid = self.grid.tolist()
        assert isinstance(grid, list), f"grid should be a list or an array, got {grid}"

        surf_gld_coin = pygame.Surface(self.cell_size)
        surf_gld_coin.fill((0, 255, 0))
        surf_crsd_coin = pygame.Surface(self.cell_size)
        surf_crsd_coin.fill((255, 0, 0))
        surf_empty = pygame.Surface(self.cell_size)
        surf_empty.fill((0, 0, 0))
        surf_map = pygame.Surface(self.cell_size)
        surf_map.fill((255, 255, 255))
        surf_wall = pygame.Surface(self.cell_size)
        surf_wall.fill((255, 255, 0))
        surf_agent = pygame.Surface(self.cell_size)
        surf_agent.fill((0, 0, 255))
        # load images for the gold coin, fire and the agent
        screen_w, screen_h = pygame.display.get_surface().get_size()
        # surf_gld_coin = pygame.transform.scale(pygame.image.load("img/gold_img.png"), (screen_w / 3, screen_h / 3))
        # surf_crsd_coin = pygame.transform.scale(pygame.image.load("img/fire_img.png"), (screen_w / 3, screen_h / 3))
        # surf_agent = pygame.transform.scale(pygame.image.load("img/agent_img.png"), (screen_w / 3, screen_h / 3))
        surf_button = pygame.transform.scale(pygame.image.load("img/button_img.png"), (screen_w / 6, screen_h / 6))

        for y_pos in range(self._n_rows):
            for x_pos in range(self._n_cols):
                pos = (x_pos * self.cell_size[0], y_pos * self.cell_size[1])

                if grid[y_pos][x_pos] == GLD_COIN:
                    self.window_surface.blit(surf_gld_coin, pos)
                if grid[y_pos][x_pos] == CRSD_COIN:
                    self.window_surface.blit(surf_crsd_coin, pos)
                if grid[y_pos][x_pos] == EMPTY:  # or grid[y][x] == AGENT:
                    self.window_surface.blit(surf_empty, pos)
                if grid[y_pos][x_pos] == BUTTON:
                    self.window_surface.blit(surf_button, pos)
                if grid[y_pos][x_pos] == AGENT:
                    self.window_surface.blit(surf_agent, pos + (0.5, 0.5))
                if grid[y_pos][x_pos] == WALL:
                    self.window_surface.blit(surf_wall, pos)
        # draw white lines between cells
        self._draw_white_lines()

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))
        else:
            raise NotImplementedError

    def _render_text(self):
        grid = self._grid.tolist()
        outfile = StringIO()

        grid = [[INT_TO_ANSI[c].decode("utf-8") for c in line] for line in grid]
        grid[self._agent_pos[0]][self._agent_pos[1]] = gym.utils.colorize(
            grid[self._agent_pos[0]][self._agent_pos[1]], "red", highlight=True
        )
        if self._last_action is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self._last_action]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in grid) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def _draw_white_lines(self):
        # white lines between cells
        w_c = (255, 255, 255)
        cell_x, cell_y = self.cell_size[0], self.cell_size[1]
        # horizontal lines
        for i in range(1, self._n_rows):
            pygame.draw.line(self.window_surface, w_c, (0, i * cell_x), (self.cell_size[0] * cell_x, i * cell_y), 3)
        # vertical lines
        for j in range(1, self._n_cols):
            pygame.draw.line(self.window_surface, w_c, (j * cell_y, 0), (j * cell_x, self.cell_size[1] * cell_y), 3)

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()
