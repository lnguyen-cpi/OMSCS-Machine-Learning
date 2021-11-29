from __future__ import annotations
import numpy as np
import random
import matplotlib.pyplot as plt
from hiive_local.mdptoolbox.openai import OpenAI_MDPToolbox
from hiive_local.mdptoolbox.mdp import ValueIteration, PolicyIteration

SEED = 42
PROBLEM_SIZE = 16

random.seed(SEED)

gamma_l = np.arange(0.1, 1, 0.1)
ave_reward = []
runtime_list = []
error_list = []
for gamma in gamma_l:
    env = OpenAI_MDPToolbox("FrozenLake-v1")
    fh = ValueIteration(
        env.P,
        env.R,
        gamma,
        # initial_value=[random.randint(0, 3) for _ in range((PROBLEM_SIZE))],
        max_iter=10000,
        epsilon=0.001
    )
    stats = fh.run()
    ave_reward.append(stats[-1]["Reward"])
    runtime_list.append(stats[-1]["Time"])
    error_list.append(stats[-1]["Error"])
    print(f"gamma: {gamma}, policy: {fh.policy}")


plt.plot(gamma_l, ave_reward, label="max reward")
plt.legend()
plt.title(f"Gamma vs Max Reward")
plt.savefig("Gamma vs Max Reward.png")

plt.clf()

plt.plot(gamma_l, runtime_list, label="runtime")
plt.xlabel("gamma")
plt.ylabel("seconds")
plt.legend()
plt.title(f"Gamma vs Runtime")
plt.savefig("Gamma vs Runtime.png")

plt.clf()

plt.plot(gamma_l, error_list, label="Error")
plt.legend()
plt.title(f"Gamma vs Error")
plt.savefig("Gamma vs Error.png")

plt.clf()


ave_reward = []
runtime_list = []
error_list = []
for gamma in gamma_l:
    env = OpenAI_MDPToolbox("FrozenLake-v1")
    fh = PolicyIteration(env.P, env.R, gamma, max_iter=10000)
    stats = fh.run()
    ave_reward.append(stats[-1]["Reward"])
    runtime_list.append(stats[-1]["Time"])
    error_list.append(stats[-1]["Error"])
    print(f"gamma: {gamma}, policy: {fh.policy}")


plt.plot(gamma_l, ave_reward, label="max reward")
plt.legend()
plt.title(f"Gamma vs Max Reward")
plt.savefig("Gamma vs Max Reward (Policy).png")

plt.clf()

plt.plot(gamma_l, runtime_list, label="runtime")
plt.xlabel("gamma")
plt.ylabel("seconds")
plt.legend()
plt.title(f"Gamma vs Runtime")
plt.savefig("Gamma vs Runtime (Policy).png")

plt.clf()

plt.plot(gamma_l, error_list, label="Error")
plt.legend()
plt.title(f"Gamma vs Error")
plt.savefig("Gamma vs Error (Policy).png")
