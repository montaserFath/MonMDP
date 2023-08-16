from contextlib import closing
import numpy as np
import gymnasium as gym
from typing import Optional
from io import StringIO

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

EMPTY = 0
AGENT = 1
GLD_COIN = 2
CRSD_COIN = 3
QCKSND = 4
QCKSND_AGNT = 5
MAP = 6

INT_TO_ANSI = {
    EMPTY: b"E",
    AGENT: b"A",
    GLD_COIN: b"G",
    CRSD_COIN: b"C",
    QCKSND: b"Q",
    QCKSND_AGNT: b"X",
    MAP: b"M",
}

GRIDS = {
    "4x8": [
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, CRSD_COIN, GLD_COIN, CRSD_COIN, EMPTY, GLD_COIN, EMPTY, EMPTY],
        [EMPTY, CRSD_COIN, EMPTY, EMPTY, EMPTY, CRSD_COIN, GLD_COIN, EMPTY],
        [EMPTY, EMPTY, EMPTY, QCKSND, MAP, EMPTY, EMPTY, EMPTY],
    ],
    "3x3": [[EMPTY, EMPTY, GLD_COIN], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]],
}


def _move(row, col, a, nrow, ncol):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    else:
        raise ValueError("illegal action")
    return (row, col)


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

    def __init__(
        self,
        render_mode: Optional[str] = None,
        grid: Optional[str] = "4x8",
        enable_quicksand: Optional[bool] = False,
        enable_map: Optional[bool] = False,
        init_agent_pos: Optional[tuple] = None,
        location_random: Optional[bool] = False,
        **kwargs,
    ):
        self._enable_quicksand = enable_quicksand
        self._enable_map = enable_map
        self._location_random = location_random
        self._grid_key = grid
        self._grid = np.asarray(GRIDS[self._grid_key])
        self._is_quicksand = QCKSND in self._grid.flatten()
        self._is_map = MAP in self._grid.flatten()

        self._n_rows, self._n_cols = self._grid.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=6, shape=(self._n_rows * self._n_cols,), dtype=int
        )
        self.action_space = gym.spaces.Discrete(
            5 if self._is_quicksand or self._is_map else 4
        )
        self._init_agent_pos = init_agent_pos
        self._agent_pos = None
        self._last_action = None
        self._has_map = None

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (min(64 * self._n_cols, 512), min(64 * self._n_rows, 512))
        self.cell_size = (
            self.window_size[0] // self._n_cols,
            self.window_size[1] // self._n_rows,
        )
        self._current_timestep = 0
        self.info = {}

    @property
    def grid(self):
        if self._has_map:
            return self._grid

        grid = self._grid.copy()
        grid[grid == GLD_COIN] = EMPTY
        grid[grid == CRSD_COIN] = EMPTY

        return grid

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self._grid = np.asarray(GRIDS[self._grid_key])
        if self._enable_map:
            self._has_map = False
        else:
            self._has_map = True
            self._grid[self._grid == MAP] = EMPTY

        if not self._enable_quicksand:
            self._grid[self._grid == QCKSND] = EMPTY
            self._agent_pos = (
                (self.np_random.integers(self._n_rows), 0)
                if self._init_agent_pos is None
                else self._init_agent_pos
            )
        self._grid[self._agent_pos] = AGENT
        self._last_action = None
        self._current_timestep = 0
        self.info = {}
        return self.grid.flatten(), {}

    def step(self, action: int):
        add_random = (
            True if self._location_random and self.np_random.random() < 0.05 else False
        )
        sand_map_env = self._is_quicksand or self._is_map
        if self._grid[self._agent_pos] != QCKSND_AGNT or add_random or not sand_map_env:
            if self._grid[self._agent_pos] == QCKSND_AGNT:
                self._grid[self._agent_pos] = QCKSND
            else:
                self._grid[self._agent_pos] = EMPTY
            self._agent_pos = _move(
                self._agent_pos[0],
                self._agent_pos[1],
                action,
                self._n_rows,
                self._n_cols,
            )
            reward = self.reward()
        else:
            self._current_timestep += 1
            return self.grid.flatten(), self.reward(), False, False, {}

        if self._grid[self._agent_pos] == MAP:
            self._grid[self._agent_pos] = AGENT
            self._has_map = True

        if self._grid[self._agent_pos] == GLD_COIN and self._has_map:
            self._grid[self._agent_pos] = AGENT
        elif self._grid[self._agent_pos] == CRSD_COIN and self._has_map:
            self._grid[self._agent_pos] = AGENT

        if self._grid[self._agent_pos] == QCKSND:
            self._grid[self._agent_pos] = QCKSND_AGNT
        else:
            self._grid[self._agent_pos] = AGENT

        terminated = (self._grid != GLD_COIN).all()

        self._last_action = action
        self._current_timestep += 1
        self.info["agent_pos"] = self._agent_pos
        return self.grid.flatten(), reward, terminated, False, self.info

    def reward(self):
        """
        Reward function returns 1 if the agent collects gold coin, -1 if the agent collects cursed coin, otherwise 0
        """
        if self._grid[self._agent_pos] == GLD_COIN and self._has_map:
            return 1.0
        if self._grid[self._agent_pos] == CRSD_COIN and self._has_map:
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
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Treasure Hunt")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

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
        surf_qcksnd = pygame.Surface(self.cell_size)
        surf_qcksnd.fill((204, 102, 0))

        for y in range(self._n_rows):
            for x in range(self._n_cols):
                pos = (x * self.cell_size[0], y * self.cell_size[1])

                if grid[y][x] == GLD_COIN:
                    self.window_surface.blit(surf_gld_coin, pos)
                elif grid[y][x] == CRSD_COIN:
                    self.window_surface.blit(surf_crsd_coin, pos)
                elif grid[y][x] == EMPTY or grid[y][x] == AGENT:
                    self.window_surface.blit(surf_empty, pos)
                elif grid[y][x] == MAP:
                    self.window_surface.blit(surf_map, pos)
                elif grid[y][x] == QCKSND or grid[y][x] == QCKSND_AGNT:
                    self.window_surface.blit(surf_qcksnd, pos)

                if grid[y][x] == AGENT or grid[y][x] == QCKSND_AGNT:
                    pos = (
                        x * self.cell_size[0] + self.cell_size[0] / 2,
                        y * self.cell_size[1] + self.cell_size[1] / 2,
                    )
                    pygame.draw.circle(
                        self.window_surface, (0, 0, 255), pos, self.cell_size[0] / 2.2
                    )

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
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

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
