from contextlib import closing
import numpy as np
import gymnasium as gym
from typing import Optional
from io import StringIO

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
COLLECT = 4

EMPTY = 0
AGENT = 1
GLD_COIN = 2
CRSD_COIN = 3
QCKSND = 4
MAP = 5

INT_TO_ANSI = {
    EMPTY: b'E',
    AGENT: b'A',
    GLD_COIN: b'G',
    CRSD_COIN: b'C',
    QCKSND: b'Q',
    MAP: b'M',
}

GRIDS = {
    "4x8":
    [
        [EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    EMPTY],
        [EMPTY,  CRSD_COIN, GLD_COIN, CRSD_COIN, QCKSND, GLD_COIN,  EMPTY,    EMPTY],
        [EMPTY,  CRSD_COIN, QCKSND,   EMPTY,     EMPTY,  CRSD_COIN, GLD_COIN, EMPTY],
        [EMPTY,  EMPTY,     EMPTY,    EMPTY,     EMPTY,  EMPTY,     EMPTY,    EMPTY],
    ],
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
    elif a == COLLECT:
        pass
    else:
        raise ValueError('illegal action')
    return (row, col)


class TreasureHunt(gym.Env):
    """
    Gridworld where the agent has to find golden coins while ignoring cursed coins.

    ## Versions
    - TreasureHunt-v0: coins are always visible to the agent.
    - TreasureHunt-v1: some cells have quicksand, where the agent gets
      stuck and has only 1% chance of escaping.
    - TreasureHunt-v2: in addition to the presence of quicksand, the agent must
      first find a map to see and collect coins.

    ## Grid
    The grid is defined by a 2D array of integers. It is possible to define
    custom grids.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 4}`.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    - 4: Collect

    If the agent is in a quicksand cell any action will fail with 99% probability.
    In TreasureHunt-v2, if the agent hasn't collected the map it can't collect
    coins.

    ## Observation Space
    The observation shape is `(rows * cols, )` denoting the flattened grid.
    Each element of the observation is in the range `{0, 5}` denoting the
    content of a cell.

    - 0: Empty
    - 1: Agent
    - 2: Golden coin (reward)
    - 3: Cursed coin (penalty)
    - 4: Quicksand
    - 5: Map

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
        1. The length of the episode is 100 for TreasureHunt-v0.
        2. The length of the episode is 200 for TreasureHunt-v1.
        3. The length of the episode is 500 for TreasureHunt-v2.

    """
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 grid: Optional[str] = "4x8",
                 enable_QCKSND: Optional[bool] = False,
                 enable_map: Optional[bool] = False,
                 **kwargs):
        self._enable_QCKSND = enable_QCKSND
        self._enable_map = enable_map
        self._grid_key = grid
        self._grid = np.asarray(MAPS[self._grid_key])
        self._n_rows, self._n_cols = self._grid.shape
        self.observation_space = gym.spaces.Box(low=0, high=5,
            shape=(self._n_rows * self._n_cols, ),
            dtype=int
        )
        self.action_space = gym.spaces.Discrete(5)
        self._agent_pos = None
        self._last_action = None

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (min(64 * self._n_cols, 512), min(64 * self._n_rows, 512))
        self.cell_size = (
            self.window_size[0] // self._n_cols,
            self.window_size[1] // self._n_rows,
        )

    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self._grid = np.asarray(MAPS[self._grid_key])
        self._agent_pos = (self.np_random.integers(self._n_rows), 0)
        self._grid[self._agent_pos] = AGENT
        self._last_action = None

        return self._grid.flatten(), {}

    def step(self, action: int):
        self._grid[self._agent_pos] = EMPTY

        self._agent_pos = _move(
            self._agent_pos[0],
            self._agent_pos[1],
            action,
            self._n_rows,
            self._n_cols
        )

        if self._grid[self._agent_pos] == REWARD:
            reward = 1
        elif self._grid[self._agent_pos] == PENALTY:
            reward = -1
        else:
            reward = 0

        self._grid[self._agent_pos] = AGENT

        is_empty = self._grid == EMPTY
        is_empty[self._agent_pos] = True

        if is_empty.all():
            terminated = True
        else:
            terminated = False

        self._last_action = action

        return self._grid.flatten(), reward, terminated, False, {}

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
                pygame.display.set_caption("Toy Grid")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        map = self._grid.tolist()
        assert isinstance(map, list), f"map should be a list or an array, got {map}"

        surf_reward = pygame.Surface(self.cell_size)
        surf_reward.fill((0, 255, 0))
        surf_penalty = pygame.Surface(self.cell_size)
        surf_penalty.fill((255, 0, 0))
        surf_empty = pygame.Surface(self.cell_size)
        surf_empty.fill((0, 0, 0))
        surf_agent = pygame.Surface(self.cell_size)
        surf_agent.fill((0, 0, 255))

        for y in range(self._n_rows):
            for x in range(self._n_cols):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                if map[y][x] == REWARD:
                    self.window_surface.blit(surf_reward, pos)
                elif map[y][x] == PENALTY:
                    self.window_surface.blit(surf_penalty, pos)
                elif map[y][x] == EMPTY:
                    self.window_surface.blit(surf_empty, pos)
                elif map[y][x] == AGENT:
                    self.window_surface.blit(surf_agent, pos)
                else:
                    raise ValueError('unknown cell type')

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
        map = self._grid.tolist()
        outfile = StringIO()

        map = [[INT_TO_ANSI[c].decode("utf-8") for c in line] for line in map]
        map[self._agent_pos[0]][self._agent_pos[1]] = gym.utils.colorize(
            map[self._agent_pos[0]][self._agent_pos[1]],
            "red",
            highlight=True
        )
        if self._last_action is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self._last_action]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in map) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
