from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from hiive_local.mdptoolbox.openai import OpenAI_MDPToolbox
from hiive_local.mdptoolbox.mdp import QLearning
from hiive_local.mdptoolbox.example import forest


BEST_GAMMA = 0.95


def compute_avg_reward(reward_matrix: np.ndarray) -> float:
    return np.average(reward_matrix)


alpha_list = np.arange(0.05, 1, 0.05)
epsilon_list = np.arange(0.05, 1, 0.05)
max_reward = []
for alpha in alpha_list:
    env = OpenAI_MDPToolbox("FrozenLake-v1")
    fh = QLearning(env.P, env.R, BEST_GAMMA, alpha=alpha)
    stats = fh.run()
    max_reward.append(stats[-1]["Reward"])


plt.plot(alpha_list, max_reward, label="max reward")
plt.legend()
plt.title(f"Alpha vs Max Reward")
plt.savefig("Alpha vs Max Reward (Lake QLearning).png")
plt.clf()


max_reward = []
for epsilon in epsilon_list:
    env = OpenAI_MDPToolbox("FrozenLake-v1")
    fh = QLearning(env.P, env.R, BEST_GAMMA, epsilon=epsilon, n_iter=100000)
    stats = fh.run()
    max_reward.append(stats[-1]["Reward"])

plt.plot(epsilon_list, max_reward, label="max reward")
plt.legend()
plt.title(f"Epsilon vs Max Reward")
plt.savefig("Epsilon vs Max Reward (Lake QLearning).png")

plt.clf()


alpha_list = np.arange(0.05, 1, 0.05)
epsilon_list = np.arange(0.05, 1, 0.05)
max_reward = []
for alpha in alpha_list:
    P, R = forest(S=5000, r1=1000, r2=5000)
    fh = QLearning(P, R, BEST_GAMMA, alpha=alpha)
    stats = fh.run()
    max_reward.append(stats[-1]["Reward"])


plt.plot(alpha_list, max_reward, label="max reward")
plt.legend()
plt.title(f"Alpha vs Max Reward")
plt.savefig("Alpha vs Max Reward (Forest QLearning).png")
plt.clf()


max_reward = []
for epsilon in epsilon_list:
    P, R = forest(S=5000, r1=1000, r2=5000)
    fh = QLearning(P, R, BEST_GAMMA, epsilon=epsilon, n_iter=100000)
    stats = fh.run()
    max_reward.append(stats[-1]["Reward"])

plt.plot(epsilon_list, max_reward, label="max reward")
plt.legend()
plt.title(f"Epsilon vs Max Reward")
plt.savefig("Epsilon vs Max Reward (Forest QLearning).png")

plt.clf()
