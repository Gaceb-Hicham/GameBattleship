from BattleshipEnv import BattleshipEnv
from QLearningAgent import QLearningAgent
from BattleshipGUI import BattleshipGUI

if __name__ == "__main__":
    train_env = BattleshipEnv(grid_size=10)
    agent = QLearningAgent(train_env, alpha=0.1, gamma=0.95, epsilon=1.0)
    agent.learn(episodes=10)
    gui_env = BattleshipEnv(grid_size=10)
    BattleshipGUI(gui_env, agent)