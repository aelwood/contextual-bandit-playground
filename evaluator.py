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
        self.oracle = {"actions": [], "rewards": []}
        self.policy = {"actions": [], "rewards": []}

    def notify(self, played_action, obtained_reward, optimal_action, optimal_reward):
        self.policy["actions"].append(played_action)
        self.policy["rewards"].append(obtained_reward)
        self.oracle["actions"].append(optimal_action)
        self.oracle["rewards"].append(optimal_reward)

    def get_stats(self):
        return f"Average oracle reward {np.mean(self.oracle['rewards']):2.4f} vs policy {np.mean(self.policy['rewards']):2.4f}"

    def plot(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        x = np.arange(len(self.policy["actions"]))

        ax1.set_title("Played actions")
        ax1.plot(x, self.policy["actions"], label="policy")
        ax1.plot(x, self.oracle["actions"], label="oracle")
        ax1.legend()

        ax2.set_title("Gained rewards")
        ax2.plot(x, self.policy["rewards"], label="policy")
        ax2.plot(x, self.oracle["rewards"], label="oracle")

        ax2.legend()
        plt.show()
