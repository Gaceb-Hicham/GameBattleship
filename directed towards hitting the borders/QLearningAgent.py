import numpy as np
import random
import pickle
import os
import csv
from math import exp

class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=1.0, q_table_file="q_table.pkl"):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.q_table = self.load_q_table()
        self.target_stack = []
        self.state_access_counts = {}
        self.total_steps = 0
        self.last_two_hits = []
        self.direct_target_prob = 1.0

    def state_to_key(self, state):
        hits = np.sum(state == 1)
        sunk_ships = len(self.env.sunk_ships)
        return (hits, sunk_ships)

    def load_q_table(self):
        if os.path.exists(self.q_table_file) and os.path.getsize(self.q_table_file) > 0:
            with open(self.q_table_file, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_q_table(self):
        with open(self.q_table_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def prune_q_table(self):
        threshold = self.total_steps - 5000
        states_to_remove = [state for state, count in self.state_access_counts.items() if count < threshold]
        for state in states_to_remove:
            if state in self.q_table:
                del self.q_table[state]
            del self.state_access_counts[state]

    def choose_action(self, state):
        if self.target_stack and random.random() < self.direct_target_prob:
            action = self.target_stack.pop()
            return action

        state_key = self.state_to_key(state)
        self.state_access_counts[state_key] = self.total_steps
        self.total_steps += 1

        if random.random() < self.epsilon or state_key not in self.q_table:
            action = self.env.action_space.sample()
            return action
        action = max(self.q_table[state_key], key=self.q_table[state_key].get)
        return action

    def learn(self, episodes=1000):
        with open('training_metrics.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Episode', 'Total Reward', 'Steps', 'Hits', 'Sunk Ships', 'Misses', 'Hit Rate', 'Q-Table Size', 'Consecutive Hits']
            for i in range(len(self.env.ships)):
                header.append(f'Max Consecutive Hits Ship {i+1}')
            writer.writerow(header)

            for ep in range(episodes):
                state, _ = self.env.reset()
                done = False
                self.target_stack.clear()
                self.last_two_hits.clear()
                total_reward = 0
                steps = 0
                hits = 0
                sunk_ships = 0
                misses = 0
                consecutive_hits_total = 0
                initial_ship_cells = self.env.remaining_ship_cells

                self.epsilon = 0.1 + 0.5 * exp(-ep / 1000)
                self.direct_target_prob = 0.1 + 0.9 * exp(-ep / 5000)
                self.alpha = max(0.01, 0.5 - ep / episodes)

                while not done:
                    state_key = self.state_to_key(state)
                    if state_key not in self.q_table:
                        self.q_table[state_key] = {a: 0 for a in range(self.env.action_space.n)}
                    self.state_access_counts[state_key] = self.total_steps

                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                    current_hits = initial_ship_cells - self.env.remaining_ship_cells
                    hits = current_hits
                    if info["sunk_ship"] is not None:
                        sunk_ships += 1
                    if info["is_miss"]:
                        misses += 1
                    consecutive_hits_total = self.env.consecutive_hits_count

                    if reward >= 5.0:
                        row, col = divmod(action, self.env.grid_size)
                        self.last_two_hits.append((row, col))
                        if len(self.last_two_hits) > 2:
                            self.last_two_hits.pop(0)

                        preferred_directions = []
                        if len(self.last_two_hits) == 2:
                            (r1, c1), (r2, c2) = self.last_two_hits
                            if r1 == r2:
                                preferred_directions = [(0, 1), (0, -1)]
                            elif c1 == c2:
                                preferred_directions = [(1, 0), (-1, 0)]

                        if preferred_directions:
                            for dr, dc in preferred_directions:
                                distance = 1
                                r, c = row + dr * distance, col + dc * distance
                                if 0 <= r < self.env.grid_size and 0 <= c < self.env.grid_size and state[r, c] == -1:
                                    self.target_stack.append(r * self.env.grid_size + c)
                        else:
                            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            for dr, dc in directions:
                                for distance in range(1, 3):
                                    r, c = row + dr * distance, col + dc * distance
                                    if 0 <= r < self.env.grid_size and 0 <= c < self.env.grid_size and state[r, c] == -1:
                                        self.target_stack.append(r * self.env.grid_size + c)

                        random.shuffle(self.target_stack)

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