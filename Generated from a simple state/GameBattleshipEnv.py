import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import tkinter as tk
import pickle
import os
import csv
from math import exp

class BattleshipEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(BattleshipEnv, self).__init__()

        self.grid_size = grid_size
        self.num_actions = grid_size * grid_size
        self.max_steps = 1000  # الحد الأقصى للخطوات في كل حلقة

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(grid_size, grid_size), dtype=np.int8)

        self._reset_board()

    def _reset_board(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.shots = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.remaining_ship_cells = 0
        self.ships = []
        self.sunk_ships = []
        self.step_count = 0  # عداد الخطوات

        ships = [2, 2, 3, 3, 4]  # أطوال السفن القياسية
        for size in ships:
            placed = False
            while not placed:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                dir = np.random.choice(['h', 'v'])
                if dir == 'h' and y + size <= self.grid_size:
                    if np.all(self.board[x, y:y+size] == 0):
                        self.board[x, y:y+size] = 1
                        self.ships.append([(x, y+i) for i in range(size)])
                        self.remaining_ship_cells += size
                        placed = True
                elif dir == 'v' and x + size <= self.grid_size:
                    if np.all(self.board[x:x+size, y] == 0):
                        self.board[x:x+size, y] = 1
                        self.ships.append([(x+i, y) for i in range(size)])
                        self.remaining_ship_cells += size
                        placed = True

        # سفينة بحجم 5x2
        placed = False
        while not placed:
            x = np.random.randint(0, self.grid_size - 1)
            y = np.random.randint(0, self.grid_size - 4)
            if np.all(self.board[x:x+2, y:y+5] == 0):
                self.board[x:x+2, y:y+5] = 1
                self.ships.append([(x+i, y+j) for i in range(2) for j in range(5)])
                self.remaining_ship_cells += 10
                placed = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_board()
        self.step_count = 0
        return self.shots.copy(), {}

    def step(self, action):
        self.step_count += 1

        row, col = divmod(action, self.grid_size)

        if self.shots[row, col] != -1:
            return self.shots.copy(), -0.2, False, False, {}  # عقوبة تكرار الخلية

        if self.board[row, col] == 1:
            self.shots[row, col] = 1
            self.remaining_ship_cells -= 1
            reward = 5.0  # مكافأة للإصابة
        else:
            self.shots[row, col] = 0
            reward = -0.1  # عقوبة أكبر للإخفاق

        sunk_ship = None
        for ship in self.ships:
            if (row, col) in ship:
                if all(self.shots[r, c] == 1 for r, c in ship):
                    sunk_ship = ship
                    self.sunk_ships.append(ship)
                    reward += 10.0  # مكافأة إضافية عند إغراق سفينة
                break

        # عقوبة صغيرة لكل خطوة
        reward -= 0.01

        terminated = self.remaining_ship_cells == 0
        truncated = self.step_count >= self.max_steps

        if terminated:
            # مكافأة إضافية بناءً على عدد الخطوات
            efficiency_bonus = 100.0 / self.step_count if self.step_count > 0 else 0
            reward += efficiency_bonus
        elif truncated:
            reward -= 10.0  # عقوبة عند الوصول إلى الحد الأقصى للخطوات

        return self.shots.copy(), reward, terminated, truncated, {"sunk_ship": sunk_ship}

    def render(self):
        print("Shots (Agent View):")
        print(self.shots)

class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=1.0, q_table_file="q_table.pkl"):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()
        self.target_stack = []
        self.state_access_counts = {}  # تتبع عدد مرات الوصول إلى كل حالة
        self.total_steps = 0  # عداد إجمالي للخطوات

    def state_to_key(self, state):
        hits = np.sum(state == 1)
        sunk_ships = len(self.env.sunk_ships)
        return (hits, sunk_ships)  # تبسيط مفتاح الحالة

    def load_q_table(self):
        if os.path.exists(self.q_table_file) and os.path.getsize(self.q_table_file) > 0:
            with open(self.q_table_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def prune_q_table(self):
        # إزالة الحالات التي لم تُستخدم منذ 5000 خطوة
        threshold = self.total_steps - 5000
        states_to_remove = [state for state, count in self.state_access_counts.items() if count < threshold]
        for state in states_to_remove:
            if state in self.q_table:
                del self.q_table[state]
            del self.state_access_counts[state]

    def choose_action(self, state):
        if self.target_stack:
            return self.target_stack.pop()

        state_key = self.state_to_key(state)
        self.state_access_counts[state_key] = self.total_steps  # تحديث عداد الوصول
        self.total_steps += 1

        if random.random() < self.epsilon or state_key not in self.q_table:
            return self.env.action_space.sample()
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def learn(self, episodes=10000):
        with open('training_metrics.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Total Reward', 'Steps', 'Hits', 'Misses', 'Hit Rate', 'Q-Table Size'])
            for ep in range(episodes):
                state, _ = self.env.reset()
                done = False
                self.target_stack.clear()
                total_reward = 0
                steps = 0
                hits = 0
                misses = 0

                # تقليل epsilon باستخدام معادلة أسية
                self.epsilon = 0.01 + 0.99 * exp(-ep / 2000)
                self.alpha = max(0.01, 0.5 - ep / episodes)

                while not done:
                    state_key = self.state_to_key(state)
                    if state_key not in self.q_table:
                        self.q_table[state_key] = {a: 0 for a in range(self.env.action_space.n)}
                    self.state_access_counts[state_key] = self.total_steps

                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                    if reward >= 5.0:  # إذا كانت المكافأة 5.0 أو أكثر (إصابة أو إغراق)
                        hits += 1
                        row, col = divmod(action, self.env.grid_size)
                        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        for dr, dc in directions:
                            r, c = row + dr, col + dc
                            if 0 <= r < self.env.grid_size and 0 <= c < self.env.grid_size and state[r, c] == -1:
                                self.target_stack.append(r * self.env.grid_size + c)
                        random.shuffle(self.target_stack)
                    elif reward == -0.1:
                        misses += 1

                    total_reward += reward
                    steps += 1

                    next_key = self.state_to_key(next_state)
                    if next_key not in self.q_table:
                        self.q_table[next_key] = {a: 0 for a in range(self.env.action_space.n)}
                    self.state_access_counts[next_key] = self.total_steps

                    best_next_action = max(self.q_table[next_key], key=self.q_table[next_key].get)
                    td_target = reward + self.gamma * self.q_table[next_key][best_next_action]
                    td_error = td_target - self.q_table[state_key][action]
                    self.q_table[state_key][action] += self.alpha * td_error

                    state = next_state
                    done = terminated or truncated

                hit_rate = hits / steps if steps > 0 else 0
                writer.writerow([ep+1, total_reward, steps, hits, misses, hit_rate, len(self.q_table)])
                print(f"Episode {ep+1}/{episodes}, Total Reward: {total_reward}, Steps: {steps}, Hits: {hits}, Misses: {misses}, Hit Rate: {hit_rate:.2%}, Q-Table Size: {len(self.q_table)}")

                # تقليم Q-Table كل 1000 حلقة
                if (ep + 1) % 1000 == 0:
                    self.prune_q_table()
                    self.save_q_table()

        # الحفظ النهائي بعد انتهاء التدريب
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

        for r in range(env.grid_size):
            self.window.grid_rowconfigure(r, weight=1, uniform="grid")
            row = []
            for c in range(env.grid_size):
                self.window.grid_columnconfigure(c, weight=1, uniform="grid")
                btn = tk.Button(self.window, text=" ", width=4, height=2, font=("Arial", 10),
                                command=lambda r=r, c=c: self.handle_click(r, c))
                btn.grid(row=r, column=c, sticky="nsew")
                row.append(btn)
            self.buttons.append(row)

        self.status_label = tk.Label(self.window, text="Your move")
        self.status_label.grid(row=env.grid_size, column=0, columnspan=env.grid_size)

        self.play_agent_button = tk.Button(self.window, text="Play Against Agent", command=self.play_vs_agent)
        self.play_agent_button.grid(row=env.grid_size+1, column=0, columnspan=env.grid_size//3)

        self.play_turns_button = tk.Button(self.window, text="Alternate Turns With Agent", command=self.play_alternative_turns)
        self.play_turns_button.grid(row=env.grid_size+1, column=env.grid_size//3, columnspan=env.grid_size//3)

        self.reset_button = tk.Button(self.window, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=env.grid_size+1, column=2*env.grid_size//3, columnspan=env.grid_size//3)

        self.restart_button = tk.Button(self.window, text="Restart Game", command=self.reset_game)
        self.restart_button.grid(row=env.grid_size+2, column=0, columnspan=env.grid_size)
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
        elif reward >= 5.0:
            self.status_label.config(text="Hit! Continue playing")
            if info.get("sunk_ship"):
                self.status_label.config(text=f"You sunk a ship!")
        elif reward == -0.1:
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

        if self.vs_agent and self.alternate_turns and reward < 5.0:
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
            if reward < 5.0:
                self.player_turn = True
                self.status_label.config(text="Your move")
            else:
                self.window.after(500, self.agent_step)
        else:
            self.window.after(500, self.agent_step)

if __name__ == "__main__":
    train_env = BattleshipEnv(grid_size=10)
    agent = QLearningAgent(train_env, alpha=0.5, gamma=0.95, epsilon=1.0)
    agent.learn(episodes=10000)

    gui_env = BattleshipEnv(grid_size=10)
    BattleshipGUI(gui_env, agent)