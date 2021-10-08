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


class Evaluator():
    def __init__(self):
        self.oracle = {'actions':[], 'rewards':[]}
        self.policy = {'actions':[], 'rewards':[]}

    def notify(self,played_action,obtained_reward, optimal_action, optimal_reward):
        self.policy['actions'].append(played_action)
        self.policy['rewards'].append(obtained_reward)
        self.oracle['actions'].append(optimal_action)
        self.oracle['rewards'].append(optimal_reward)

    def get_stats(self):
        return f"Average oracle reward {np.mean(self.oracle['rewards']):2.4f} vs policy {np.mean(self.policy['rewards']):2.4f}"
