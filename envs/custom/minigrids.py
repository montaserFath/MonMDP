from minigrid.envs.lavagap import LavaGapEnv


class LavaCrossingNoDeath(LavaGapEnv):
    """
    Version of MiniGrid LavaCrossing where lava cells do not kill the agent.
    Instead, they yield a penalty of -10.

    """
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        current_pos = self.front_pos
        current_cell = self.grid.get(*current_pos)
        if current_cell is not None and current_cell.type == "lava":
            terminated = False
            reward = -10

        return obs, reward, terminated, truncated, info
