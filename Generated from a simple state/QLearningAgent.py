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
