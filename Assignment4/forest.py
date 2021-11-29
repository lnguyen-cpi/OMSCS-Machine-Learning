from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from hiive_local.mdptoolbox.mdp import ValueIteration, PolicyIteration
from hiive_local.mdptoolbox.example import forest


def compute_avg_reward(reward_matrix: np.ndarray) -> float:
    return np.average(reward_matrix)


gamma_l = np.arange(0.05, 1, 0.05)
max_reward = []
runtime_list = []
error_list = []
for gamma in gamma_l:
    P, R = forest(S=1000, r1=1000, r2=5000)
    fh = ValueIteration(P, R, gamma)
    stats = fh.run()
    max_reward.append(stats[-1]["Reward"])
    runtime_list.append(stats[-1]["Time"])
    error_list.append(stats[-1]["Error"])
    print(f"gamma: {gamma}, policy: {fh.policy}")
    plt.clf()
    plt.plot(range(len(fh.policy)), fh.policy, label="Action (0=wait, 1=Cut)")
    plt.legend()
    plt.title(f"Policy with gamma {round(gamma, 3)}")
    plt.savefig(f"Policy with gamma {round(gamma, 3)} (Forest).png")


plt.plot(gamma_l, max_reward, label="max reward")
plt.legend()
plt.title(f"Gamma vs Max Reward")
plt.savefig("Gamma vs Max Reward (Forest).png")

plt.clf()

plt.plot(gamma_l, runtime_list, label="runtime")
plt.legend()
plt.title(f"Gamma vs Runtime")
plt.savefig("Gamma vs Runtime (Forest).png")

plt.clf()

plt.plot(gamma_l, error_list, label="Error")
plt.legend()
plt.title(f"Gamma vs Error")
plt.savefig("Gamma vs Error (Forest).png")

plt.clf()


gamma_l = np.arange(0.05, 1, 0.05)
max_reward = []
runtime_list = []
error_list = []
action_list = []
for gamma in gamma_l:
    P, R = forest(S=5000, r1=1000, r2=5000)
    fh = PolicyIteration(P, R, gamma)
    stats = fh.run()
    max_reward.append(stats[-1]["Reward"])
    runtime_list.append(stats[-1]["Time"])
    error_list.append(stats[-1]["Error"])
    action_list.append(fh.policy)
    print(f"gamma: {gamma}, policy: {fh.policy}")
    plt.clf()
    plt.plot(range(len(fh.policy)), fh.policy, label="Action (0=wait, 1=Cut)")
    plt.legend()
    plt.title(f"Policy with gamma {round(gamma, 3)}")
    plt.savefig(f"Policy with gamma {round(gamma, 3)} ((Forest Policy).png")


plt.plot(gamma_l, max_reward, label="max reward")
plt.legend()
plt.title(f"Gamma vs Max Reward")
plt.savefig("Gamma vs Max Reward ((Forest Policy).png")

plt.clf()

plt.plot(gamma_l, runtime_list, label="runtime")
plt.xlabel("gamma")
plt.ylabel("seconds")
plt.legend()
plt.title(f"Gamma vs Runtime")
plt.savefig("Gamma vs Runtime ((Forest Policy).png")

plt.clf()

plt.plot(gamma_l, error_list, label="Error")
plt.legend()
plt.title(f"Gamma vs Error")
plt.savefig("Gamma vs Error ((Forest Policy).png")


