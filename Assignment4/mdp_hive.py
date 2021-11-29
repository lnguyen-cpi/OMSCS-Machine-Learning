from __future__ import annotations
import mdptoolbox.example
import numpy as np
import matplotlib.pyplot as plt
from hiive_local.mdptoolbox.openai import OpenAI_MDPToolbox
from hiive_local.mdptoolbox.mdp import ValueIteration, PolicyIteration


def compute_avg_reward(reward_matrix: np.ndarray) -> float:
    return np.average(reward_matrix)

# env = OpenAIMDPToolbox("FrozenLake-v1", render=True)
# fh = mdptoolbox.mdp.ValueIteration(env.P, env.R, 0.95)
# fh.run()
# print(fh.V)
# print(fh.policy)


# gamma_l = np.arange(0.05, 1, 0.05)
# ave_reward = []
# runtime_list = []
# error_list = []
# for gamma in gamma_l:
#     env = OpenAI_MDPToolbox("FrozenLake-v1", render=True)
#     fh = ValueIteration(env.P, env.R, gamma)
#     stats = fh.run()
#     ave_reward.append(
#         compute_avg_reward(fh.V)
#     )
#     runtime_list.append(fh.time)
#     error_list.append(stats[-1]["Error"])
#
#
# plt.plot(gamma_l, ave_reward, label="average reward")
# plt.legend()
# plt.title(f"Gamma vs Average Reward")
# plt.savefig("Gamma vs Average Reward.png")
#
# plt.clf()
#
# plt.plot(gamma_l, runtime_list, label="runtime")
# plt.legend()
# plt.title(f"Gamma vs Runtime")
# plt.savefig("Gamma vs Runtime.png")
#
# plt.clf()
#
# plt.plot(gamma_l, error_list, label="Error")
# plt.legend()
# plt.title(f"Gamma vs Error")
# plt.savefig("Gamma vs Error.png")


gamma_l = np.arange(0.05, 1, 0.05)
ave_reward = []
runtime_list = []
error_list = []
for gamma in gamma_l:
    env = OpenAI_MDPToolbox("FrozenLake-v1", render=True)
    fh = PolicyIteration(env.P, env.R, gamma)
    stats = fh.run()
    ave_reward.append(
        compute_avg_reward(fh.V)
    )
    runtime_list.append(fh.time)
    error_list.append(stats[-1]["Error"])


plt.plot(gamma_l, ave_reward, label="average reward")
plt.legend()
plt.title(f"Gamma vs Average Reward")
plt.savefig("Gamma vs Average Reward (Policy).png")

plt.clf()

plt.plot(gamma_l, runtime_list, label="runtime")
plt.legend()
plt.title(f"Gamma vs Runtime")
plt.savefig("Gamma vs Runtime (Policy).png")

plt.clf()

plt.plot(gamma_l, error_list, label="Error")
plt.legend()
plt.title(f"Gamma vs Error")
plt.savefig("Gamma vs Error (Policy).png")
