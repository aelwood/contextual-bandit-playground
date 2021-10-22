from __future__ import annotations

import abc
import typing
import scipy
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC
from scipy.stats import norm

if typing.TYPE_CHECKING:
    from typing import Any, Iterable, NamedTuple, Optional, Sequence, Type, Union


class Evaluator:
    def __init__(self):
        self.oracle = {"actions": [], "rewards": [], "stochastic_reward": []}
        self.policy = {"actions": [], "rewards": [], "stochastic_reward": []}
    # (a, r,s_r, optimal_a, optimal_r,stochastic_r)
    def notify(self, played_action, obtained_reward,obtained_stochastic_reward, optimal_action, optimal_reward, stochastic_reward):
        self.policy["actions"].append(played_action)
        self.policy["rewards"].append(obtained_reward)
        self.policy["stochastic_reward"].append(obtained_stochastic_reward)
        self.oracle["actions"].append(optimal_action)
        self.oracle["rewards"].append(optimal_reward)
        self.oracle["stochastic_reward"].append(stochastic_reward)

    def get_stats(self):
        return f"Average oracle reward {np.mean(self.oracle['rewards']):2.4f} vs policy {np.mean(self.policy['rewards']):2.4f}"

    def plot(self):

        plt.figure(figsize=(16,9))


        x = np.arange(len(self.policy["actions"]))


        ax1 = plt.subplot(211)

        ax1.set_title("Played actions")
        ax1.plot(x, self.policy["actions"], label="policy",linestyle="",marker="^")
        ax1.plot(x, self.oracle["actions"], label="oracle",linestyle="",marker="x")


        ax2 = plt.subplot(223)
        ax2.set_title("Cumulative Rewards")
        ax2.plot(x, np.cumsum(self.policy["rewards"]), label="policy(expected)", alpha=0.6)
        ax2.plot(x, np.cumsum(self.oracle["rewards"]), label="oracle(expected)", alpha=0.6)
        ax2.plot(x, np.cumsum(self.policy["stochastic_reward"]), label="policy(stochastic)", alpha=0.6)
        ax2.plot(x, np.cumsum(self.oracle["stochastic_reward"]), label="oracle(stochastic)", alpha=0.6)
        ax2.legend()


        ax3 = plt.subplot(224)
        ax3.set_title("Cumulative Regrets")
        ax3.plot(x, np.cumsum(self.oracle["stochastic_reward"])-np.cumsum(self.policy["stochastic_reward"]), label="stochastic_Regrets", lw=4, alpha=0.8)
        ax3.plot(x, np.cumsum(self.oracle["rewards"])-np.cumsum(self.policy["rewards"]), label="Expected_Regrets", lw=4, alpha=0.8)

        ax3.legend()




        ax1.legend()
        plt.show()