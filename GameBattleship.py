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

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, q_table_file="q_table_improved.pkl"):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()
        self.total_steps = 0
        self.state_access_counts = {}
        self.memory = deque(maxlen=10000)

    def state_to_key(self, state):
        hits = np.sum(state == 1)
        misses = np.sum(state == 0)
        sunk_ships = len(self.env.sunk_ships)
        grid_size = self.env.grid_size
        hit_regions = (
            np.sum(state[:grid_size//2, :grid_size//2] == 1),
            np.sum(state[:grid_size//2, grid_size//2:] == 1),
            np.sum(state[grid_size//2:, :grid_size//2] == 1),
            np.sum(state[grid_size//2:, grid_size//2:] == 1)
        )
        return (hits, misses, sunk_ships, hit_regions)

    def load_q_table(self):
        if os.path.exists(self.q_table_file) and os.path.getsize(self.q_table_file) > 0:
            with open(self.q_table_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def prune_q_table(self):
        threshold = self.total_steps - 2000
        states_to_remove = [state for state, count in self.state_access_counts.items() if count < threshold]
        for state in states_to_remove:
            if state in self.q_table:
                del self.q_table[state]
            del self.state_access_counts[state]

    def choose_action(self, state):
        state_key = self.state_to_key(state)
        self.state_access_counts[state_key] = self.total_steps
        self.total_steps += 1
        if random.random() < self.epsilon or state_key not in self.q_table:
            return self.env.action_space.sample()
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_key = self.state_to_key(state)
            next_key = self.state_to_key(next_state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0 for a in range(self.env.action_space.n)}
            if next_key not in self.q_table:
                self.q_table[next_key] = {a: 0 for a in range(self.env.action_space.n)}
            target = reward if done else reward + self.gamma * max(self.q_table[next_key].values())
            td_error = target - self.q_table[state_key][action]
            self.q_table[state_key][action] += self.alpha * td_error

    def learn(self, episodes=20000):
        with open('training_metrics_improved.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # إضافة أعمدة لكل سفينة لتسجيل أقصى إصابات متتابعة
            header = ['Episode', 'Total Reward', 'Steps', 'Hits', 'Sunk Ships', 'Misses', 'Hit Rate', 'Q-Table Size', 'Consecutive Hits']
            for i in range(len(self.env.ships)):
                header.append(f'Max Consecutive Hits Ship {i+1}')
            writer.writerow(header)

            for ep in range(episodes):
                state, _ = self.env.reset()
                done = False
                total_reward = 0
                steps = 0
                hits = 0
                sunk_ships = 0
                misses = 0
                consecutive_hits_total = 0
                initial_ship_cells = self.env.remaining_ship_cells

                self.epsilon = max(0.01, 1.0 * np.exp(-ep / 5000))

                while not done:
                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.store_experience(state, action, reward, next_state, terminated or truncated)

                    current_hits = initial_ship_cells - self.env.remaining_ship_cells
                    hits = current_hits
                    if info["sunk_ship"] is not None:
                        sunk_ships += 1
                    if reward < 0 and reward >= -0.05:
                        misses += 1
                    consecutive_hits_total = self.env.consecutive_hits_count

                    total_reward += reward
                    steps += 1
                    state = next_state
                    done = terminated or truncated

                    self.replay()

                hit_rate = hits / steps if steps > 0 else 0
                # جمع أقصى إصابات متتابعة لكل سفينة
                row = [ep+1, total_reward, steps, hits, sunk_ships, misses, hit_rate, len(self.q_table), consecutive_hits_total]
                for ship in self.env.ships:
                    ship_key = tuple(ship)
                    row.append(self.env.max_consecutive_hits[ship_key])
                writer.writerow(row)
                print(f"Episode {ep+1}/{episodes}, Total Reward: {total_reward}, Steps: {steps}, Hit Rate: {hit_rate:.2%}, Consecutive Hits: {consecutive_hits_total}")

                if (ep + 1) % 1000 == 0:
                    self.prune_q_table()
                    self.save_q_table()

        self.save_q_table()

class BattleshipGUI:
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        self.window = tk.Tk()
        self.window.title("Battleship Game")
        self.buttons = []
        self.state, _ = self.env.reset()
        self.done = False
        self.vs_agent = False
        self.alternate_turns = False
        self.player_turn = True
        self.destroyed_cells = set()

        tk.Label(self.window, text="Player Name:").grid(row=self.env.grid_size+3, column=0, columnspan=2)
        self.player_name = tk.Entry(self.window)
        self.player_name.grid(row=self.env.grid_size+3, column=2, columnspan=self.env.grid_size-2)
        self.player_name.insert(0, "Player")

        for r in range(self.env.grid_size):
            self.window.grid_rowconfigure(r, weight=1, uniform="grid")
            row = []
            for c in range(self.env.grid_size):
                self.window.grid_columnconfigure(c, weight=1, uniform="grid")
                btn = tk.Button(self.window, text=" ", width=4, height=2, font=("Arial", 10),
                                command=lambda r=r, c=c: self.handle_click(r, c))
                btn.grid(row=r, column=c, sticky="nsew")
                row.append(btn)
            self.buttons.append(row)

        self.status_label = tk.Label(self.window, text="Your move")
        self.status_label.grid(row=self.env.grid_size, column=0, columnspan=self.env.grid_size)

        self.play_agent_button = tk.Button(self.window, text="Play Against Agent", command=self.play_vs_agent)
        self.play_agent_button.grid(row=self.env.grid_size+1, column=0, columnspan=self.env.grid_size//3)

        self.play_turns_button = tk.Button(self.window, text="Alternate Turns With Agent", command=self.play_alternative_turns)
        self.play_turns_button.grid(row=self.env.grid_size+1, column=self.env.grid_size//3, columnspan=self.env.grid_size//3)

        self.reset_button = tk.Button(self.window, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=self.env.grid_size+1, column=2*self.env.grid_size//3, columnspan=self.env.grid_size//3)

        self.restart_button = tk.Button(self.window, text="Restart Game", command=self.reset_game)
        self.restart_button.grid(row=self.env.grid_size+2, column=0, columnspan=self.env.grid_size)
        self.restart_button.grid_remove()

        self.window.mainloop()

    def handle_click(self, row, col):
        if self.done or (self.vs_agent and not self.alternate_turns) or (self.vs_agent and self.alternate_turns and not self.player_turn):
            return
        action = row * self.env.grid_size + col
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = obs
        self.update_buttons(obs, info)
        if reward == -0.2:
            self.status_label.config(text="Cell already selected!")
            return
        elif reward >= 1.0:
            self.status_label.config(text="Hit! Continue playing")
            if info.get("sunk_ship"):
                self.status_label.config(text=f"You sunk a ship!")
        elif reward <= -0.05:
            self.status_label.config(text="Miss")
        if terminated:
            player_name = self.player_name.get() or "Player"
            self.status_label.config(text=f"{player_name} Wins! All ships destroyed!")
            self.done = True
            self.restart_button.grid()
            return
        elif truncated:
            self.status_label.config(text="Game truncated: Too many steps!")
            self.done = True
            self.restart_button.grid()
            return
        if self.vs_agent and self.alternate_turns and reward < 1.0:
            self.player_turn = False
            self.window.after(500, self.agent_step)

    def update_buttons(self, obs, info=None):
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                if info and info.get("sunk_ship") and (r, c) in info["sunk_ship"]:
                    self.destroyed_cells.add((r, c))
                if (r, c) in self.destroyed_cells:
                    self.buttons[r][c].config(bg="green", text="X")
                else:
                    val = obs[r, c]
                    if val == 1:
                        self.buttons[r][c].config(bg="red", text="X")
                    elif val == 0:
                        self.buttons[r][c].config(bg="blue", text="O")
                    else:
                        self.buttons[r][c].config(bg="SystemButtonFace", text=" ")

    def reset_game(self):
        self.state, _ = self.env.reset()
        self.done = False
        self.player_turn = True
        self.destroyed_cells.clear()
        self.status_label.config(text="Your move")
        self.update_buttons(self.state)
        self.restart_button.grid_remove()
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                self.buttons[r][c].config(width=4)

    def play_vs_agent(self):
        if not self.agent:
            self.status_label.config(text="No agent loaded")
            return
        self.vs_agent = True
        self.alternate_turns = False
        self.reset_game()
        self.status_label.config(text="Agent is playing...")
        self.window.after(500, self.agent_step)

    def play_alternative_turns(self):
        if not self.agent:
            self.status_label.config(text="No agent loaded")
            return
        self.vs_agent = True
        self.alternate_turns = True
        self.reset_game()
        self.status_label.config(text="Your move - alternating turns")

    def agent_step(self):
        if self.done:
            return
        action = self.agent.choose_action(self.state)
        self.state, reward, terminated, truncated, info = self.env.step(action)
        self.update_buttons(self.state, info)
        if info.get("sunk_ship"):
            self.status_label.config(text="Agent sunk a ship!")
        if terminated:
            self.status_label.config(text="Agent Wins! All ships destroyed!")
            self.done = True
            self.restart_button.grid()
            return
        elif truncated:
            self.status_label.config(text="Agent failed: Too many steps!")
            self.done = True
            self.restart_button.grid()
            return
        if self.alternate_turns:
            if reward < 1.0:
                self.player_turn = True
                self.status_label.config(text="Your move")
            else:
                self.window.after(500, self.agent_step)
        else:
            self.window.after(500, self.agent_step)

if __name__ == "__main__":
    train_env = BattleshipEnv(grid_size=10)
    agent = QLearningAgent(train_env, alpha=0.1, gamma=0.95, epsilon=1.0)
    agent.learn(episodes=1)
    gui_env = BattleshipEnv(grid_size=10)
    BattleshipGUI(gui_env, agent)