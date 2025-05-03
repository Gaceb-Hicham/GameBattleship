import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_metrics.csv')

plt.figure(figsize=(10, 5))
plt.plot(data['Episode'], data['Total Reward'], label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig('total_reward.png')

plt.figure(figsize=(10, 5))
plt.plot(data['Episode'], data['Hit Rate'], label='Hit Rate', color='green')
plt.xlabel('Episode')
plt.ylabel('Hit Rate')
plt.title('Hit Rate Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig('hit_rate.png')

plt.figure(figsize=(10, 5))
plt.plot(data['Episode'], data['Steps'], label='Steps', color='red')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps Over Episodes')
plt.legend()
plt.grid(True)
plt.savefig('steps.png')