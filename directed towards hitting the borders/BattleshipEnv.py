import gymnasium as gym
import numpy as np
from gymnasium import spaces
from math import exp

class BattleshipEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(BattleshipEnv, self).__init__()
        self.grid_size = grid_size
        self.num_actions = grid_size * grid_size
        self.max_steps = 1000
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(grid_size, grid_size), dtype=np.int8)
        self._reset_board()

    def _reset_board(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.shots = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.remaining_ship_cells = 0
        self.ships = []
        self.sunk_ships = []
        self.step_count = 0
        self.last_hit_position = None
        self.last_hit_ship = None
        self.consecutive_hits = {}
        self.max_consecutive_hits = {}
        self.consecutive_hits_count = 0

        ships = [2, 2, 3, 3, 4]
        for size in ships:
            placed = False
            while not placed:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                dir = np.random.choice(['h', 'v'])
                if dir == 'h' and y + size <= self.grid_size:
                    if np.all(self.board[x, y:y+size] == 0):
                        self.board[x, y:y+size] = 1
                        ship = [(x, y+i) for i in range(size)]
                        self.ships.append(ship)
                        self.remaining_ship_cells += size
                        ship_key = tuple(ship)
                        self.consecutive_hits[ship_key] = 0
                        self.max_consecutive_hits[ship_key] = 0
                        placed = True
                elif dir == 'v' and x + size <= self.grid_size:
                    if np.all(self.board[x:x+size, y] == 0):
                        self.board[x:x+size, y] = 1
                        ship = [(x+i, y) for i in range(size)]
                        self.ships.append(ship)
                        self.remaining_ship_cells += size
                        ship_key = tuple(ship)
                        self.consecutive_hits[ship_key] = 0
                        self.max_consecutive_hits[ship_key] = 0
                        placed = True

        placed = False
        while not placed:
            x = np.random.randint(0, self.grid_size - 1)
            y = np.random.randint(0, self.grid_size - 4)
            if np.all(self.board[x:x+2, y:y+5] == 0):
                self.board[x:x+2, y:y+5] = 1
                ship = [(x+i, y+j) for i in range(2) for j in range(5)]
                self.ships.append(ship)
                self.remaining_ship_cells += 10
                ship_key = tuple(ship)
                self.consecutive_hits[ship_key] = 0
                self.max_consecutive_hits[ship_key] = 0
                placed = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_board()
        self.step_count = 0
        self.last_hit_position = None
        self.last_hit_ship = None
        self.consecutive_hits_count = 0
        return self.shots.copy(), {}

    def step(self, action):
        self.step_count += 1
        row, col = divmod(action, self.grid_size)
        current_position = (row, col)

        if self.shots[row, col] != -1:
            return self.shots.copy(), -0.2, False, False, {"sunk_ship": None, "is_miss": False}

        reward = 0.0
        sunk_ship = None
        is_miss = False

        if self.board[row, col] == 1:
            self.shots[row, col] = 1
            self.remaining_ship_cells -= 1
            reward = 5.0

            for ship in self.ships:
                if (row, col) in ship:
                    ship_key = tuple(ship)
                    if ship_key not in self.sunk_ships:
                        if self.last_hit_ship == ship_key:
                            self.consecutive_hits[ship_key] += 1
                            if self.consecutive_hits[ship_key] >= 2:
                                reward += 3.0
                                self.consecutive_hits_count += 1
                        else:
                            self.consecutive_hits[ship_key] = 1

                        self.max_consecutive_hits[ship_key] = max(self.max_consecutive_hits[ship_key], self.consecutive_hits[ship_key])

                        self.last_hit_ship = ship_key
                        self.last_hit_position = current_position

                        if all(self.shots[r, c] == 1 for r, c in ship):
                            sunk_ship = ship
                            self.sunk_ships.append(ship)
                            reward += 10.0
                    break
        else:
            self.shots[row, col] = 0
            reward = -0.1
            is_miss = True

        reward -= 0.01

        terminated = self.remaining_ship_cells == 0
        truncated = self.step_count >= self.max_steps

        if terminated:
            efficiency_bonus = 100.0 / self.step_count if self.step_count > 0 else 0
            reward += efficiency_bonus
        elif truncated:
            reward -= 10.0

        return self.shots.copy(), reward, terminated, truncated, {"sunk_ship": sunk_ship, "is_miss": is_miss}

    def render(self):
        print("Shots (Agent View):")
        print(self.shots)