from __future__ import annotations

import abc
import collections
import math
import numpy as np
import typing
import scipy

from abc import ABC
from scipy.stats import norm

if typing.TYPE_CHECKING:
    from typing import Any, Iterable, NamedTuple, Optional, Sequence, Type, Union, Dict


class PolicyABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def notify_event(self, context, action, reward):
        pass

    @abc.abstractmethod
    def get_action(self, context):
        pass


class RandomPolicy(PolicyABC, ABC):
    def __init__(self, distribution):
        assert type(distribution) == scipy.stats._distn_infrastructure.rv_frozen
        self.distribution = distribution

    def train(self):
        pass

    def notify_event(self, context, action, reward):
        pass

    def get_action(self, context):
        return self.distribution.rvs()


class UcbPolicy(PolicyABC, ABC):
    """
    This is the implementation of the UCB1 policy, found on this website:
    https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047

    For this solution, context is not taken into account
    arm_values = dict arm id(int) -> arm value (float)
    epsilon = scalar from which consider the reward a positive reward
    sw = sliding windows parameters: number of samples to consider during the training phase
    """

    def __init__(self, arm_values: Dict[int, float], epsilon=0.02, sw=0):
        self.arm_values = arm_values
        self.values_arm = {v: k for k, v in arm_values.items()}
        self.epsilon = epsilon
        self.arms = {
            a_id: scipy.stats.beta(a=1.0, b=1.0) for a_id in self.arm_values.keys()
        }

        sw *= -1 if sw > 0 else 1
        self.sw = sw

        self.past_rewards = []
        self.past_actions = []

    def train(self):
        alphas = collections.defaultdict(int)
        betas = collections.defaultdict(int)
        arm_ids = []
        for action, reward in zip(self.past_actions[self.sw:], self.past_rewards[self.sw:]):
            arm_id = self.values_arm[action]
            arm_ids.append(arm_id)
            is_success = reward
            if is_success:
                alphas[arm_id] += 1
            else:
                betas[arm_id] += 1

        for arm_id in np.unique(arm_ids):
            self.arms[arm_id] = scipy.stats.beta(a=alphas[arm_id]+1, b=betas[arm_id]+1)

    def notify_event(self, context, action, stochastic_reward):
        self.past_rewards.append(stochastic_reward)
        self.past_actions.append(action)

    def _get_total_pulls(self, arm_beta):
        return arm_beta.kwds["a"] + arm_beta.kwds["b"]

    def get_action(self, context):
        """
        math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
        where total_counts is the total number of trials, self.counts[arm] is the number
        that a specific arm was pulled
        """

        total_pulls = np.sum(
            [self._get_total_pulls(arm_beta) for _, arm_beta in self.arms.items()]
        )

        samples = {}
        for id, arm_beta in self.arms.items():
            num_of_pulls_from_the_arm = self._get_total_pulls(arm_beta)
            arm_mean_reward = arm_beta.mean()

            arm_value = self.arm_values[id]

            if num_of_pulls_from_the_arm == 2:
                return float(arm_value)
            upper_bound_value = arm_mean_reward + math.sqrt(
                (2 * math.log(total_pulls)) / float(num_of_pulls_from_the_arm)
            )
            samples[arm_value] = upper_bound_value
        markup_of_max = max(samples, key=samples.get)
        return float(markup_of_max)
