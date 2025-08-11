# grid_env.py

import numpy as np

class GridEnv:
    ACTIONS = ['up', 'down', 'left', 'right']
    ACTION_MAP = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(self, grid):
        """
        grid: 2D numpy array of ints (0=empty, 1=mouse, 2=wall, 3=trap, 4=cheese, 5=organic)
        """
        self.grid = np.array(grid)
        self.size = self.grid.shape[0]
        self.mouse_pos = tuple(np.argwhere(self.grid == 1)[0])
        self.done = False
        self.original_grid = self.grid.copy()

    def reset(self):
        self.grid = self.original_grid.copy()
        self.mouse_pos = tuple(np.argwhere(self.grid == 1)[0])
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns a one-hot encoded tensor: shape (6, 5, 5)
        """
        state = np.zeros((6, self.size, self.size), dtype=np.float32)
        for i in range(self.size):
            for j in range(self.size):
                elem = self.grid[i, j]
                state[elem, i, j] = 1.0
        return state

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        dx, dy = self.ACTION_MAP[action]
        x, y = self.mouse_pos
        nx, ny = x + dx, y + dy

        reward = -0.2  # default penalty

        # Check bounds
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return self.get_state(), reward, self.done

        target = self.grid[nx, ny]

        if target == 2:  # wall
            return self.get_state(), reward, self.done
        elif target == 3:  # trap
            reward = -50
            self.done = True
        elif target in [4, 5]:  # cheese
            reward = 10
            self.done = True

        # Move mouse
        self.grid[x, y] = 0  # clear old
        self.grid[nx, ny] = 1  # new pos
        self.mouse_pos = (nx, ny)

        return self.get_state(), reward, self.done

    def render(self):
        emoji = ['.', '🐭', '🧱', '☠️', '🧀', '🥦']
        print("\n".join(" ".join(emoji[self.grid[i,j]] for j in range(self.size)) for i in range(self.size)))
        print()
