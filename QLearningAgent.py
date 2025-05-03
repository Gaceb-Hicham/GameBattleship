import numpy as np
import random
import pickle
import os
import csv
from collections import deque

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