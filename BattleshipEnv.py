import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import tkinter as tk
import pickle
import os
import csv
from collections import deque

class BattleshipEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(BattleshipEnv, self).__init__()
        self.grid_size = grid_size
        self.num_actions = grid_size * grid_size
        self.max_steps = 300
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
        self.last_hit_ship = None
        self.consecutive_hits = {}
        self.max_consecutive_hits = {}  # تتبع أقصى إصابات متتابعة لكل سفينة
        self.steps_since_hit = {}
        self.last_hit_position = None
        self.misses_during_ship = {}
        self.consecutive_hits_count = 0

        # تعديل قائمة السفن: استبدال سفينة بطول 3 بطول 5، وإضافة سفينة 5x2
        ships = [
            (2, 1),  # سفينة 2x1
            (2, 1),  # سفينة 2x1
            (3, 1),  # سفينة 3x1 (تبقى كما هي)
            (5, 1),  # سفينة 5x1 (بدلاً من 3x1)
            (4, 1),  # سفينة 4x1
            (5, 2)   # سفينة جديدة 5x2
        ]

        for length, width in ships:
            placed = False
            while not placed:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                dir = np.random.choice(['h', 'v'])
                if dir == 'h':
                    if y + length <= self.grid_size and x + width <= self.grid_size:
                        if np.all(self.board[x:x+width, y:y+length] == 0):
                            self.board[x:x+width, y:y+length] = 1
                            ship = [(x + i, y + j) for i in range(width) for j in range(length)]
                            self.ships.append(ship)
                            self.remaining_ship_cells += length * width
                            ship_key = tuple(ship)
                            self.consecutive_hits[ship_key] = 0
                            self.max_consecutive_hits[ship_key] = 0
                            self.steps_since_hit[ship_key] = 0
                            self.misses_during_ship[ship_key] = 0
                            placed = True
                elif dir == 'v':
                    if x + length <= self.grid_size and y + width <= self.grid_size:
                        if np.all(self.board[x:x+length, y:y+width] == 0):
                            self.board[x:x+length, y:y+width] = 1
                            ship = [(x + j, y + i) for j in range(length) for i in range(width)]
                            self.ships.append(ship)
                            self.remaining_ship_cells += length * width
                            ship_key = tuple(ship)
                            self.consecutive_hits[ship_key] = 0
                            self.max_consecutive_hits[ship_key] = 0
                            self.steps_since_hit[ship_key] = 0
                            self.misses_during_ship[ship_key] = 0
                            placed = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_board()
        self.step_count = 0
        self.last_hit_ship = None
        self.last_hit_position = None
        self.consecutive_hits_count = 0
        return self.shots.copy(), {}

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        self.step_count += 1
        row, col = divmod(action, self.grid_size)
        current_position = (row, col)

        for ship in self.ships:
            ship_key = tuple(ship)
            if ship_key not in self.sunk_ships:
                self.steps_since_hit[ship_key] += 1

        if self.shots[row, col] != -1:
            return self.shots.copy(), -0.2, False, False, {"sunk_ship": None, "consecutive_hits": 0}

        reward = 0.0
        sunk_ship = None
        consecutive_hits_step = 0

        if self.board[row, col] == 1:
            self.shots[row, col] = 1
            self.remaining_ship_cells -= 1
            reward += 1.0

            for ship in self.ships:
                if (row, col) in ship:
                    ship_key = tuple(ship)
                    if ship_key not in self.sunk_ships:
                        if self.last_hit_ship == ship_key:
                            self.consecutive_hits[ship_key] += 1
                            if self.consecutive_hits[ship_key] >= 2:
                                reward += 2.0
                                consecutive_hits_step = self.consecutive_hits[ship_key]
                                self.consecutive_hits_count += 1
                        else:
                            self.consecutive_hits[ship_key] = 1

                        # تحديث أقصى إصابات متتابعة
                        self.max_consecutive_hits[ship_key] = max(self.max_consecutive_hits[ship_key], self.consecutive_hits[ship_key])

                        self.steps_since_hit[ship_key] = 0
                        self.last_hit_ship = ship_key
                        self.last_hit_position = current_position

                        if all(self.shots[r, c] == 1 for r, c in ship):
                            sunk_ship = ship
                            self.sunk_ships.append(ship)
                            reward += 20.0
                            if self.misses_during_ship[ship_key] == 0:
                                reward += 5.0
                    break
        else:
            self.shots[row, col] = 0
            reward -= 0.05
            if self.last_hit_ship and self.last_hit_ship not in self.sunk_ships:
                self.misses_during_ship[self.last_hit_ship] += 1

        if self.last_hit_ship and self.last_hit_ship not in self.sunk_ships:
            if self.last_hit_position and self.steps_since_hit[self.last_hit_ship] <= 5:
                distance = self._manhattan_distance(self.last_hit_position, current_position)
                if distance > 3 and self.board[row, col] != 1:
                    reward -= 1.0

        terminated = self.remaining_ship_cells == 0
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward += 100.0 / self.step_count if self.step_count > 0 else 0
        elif truncated:
            reward -= 10.0

        return self.shots.copy(), reward, terminated, truncated, {"sunk_ship": sunk_ship, "consecutive_hits": consecutive_hits_step}

    def render(self):
        print("Shots (Agent View):")
        print(self.shots)
