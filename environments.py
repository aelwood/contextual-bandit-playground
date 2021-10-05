from __future__ import annotations

import abc
import numpy as np
import typing

from abc import ABC

from sklearn.datasets import make_blobs
from scipy.stats import norm

if typing.TYPE_CHECKING:
    from typing import Any, Iterable, NamedTuple, Optional, Sequence, Type, Union


class EnvironmentABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_reward(self, action, context):
        pass

    @abc.abstractmethod
    def generate_contexts(self):
        pass

    @abc.abstractmethod
    def get_best_reward_action(self):
        pass


class SyntheticEnvironment(EnvironmentABC, ABC):
    def __init__(
        self,
        number_of_different_context: int,
        number_of_observations: int,
        time_perturbation_function,
        fixed_variance=0.1,
    ):
        """
        It creates number_of_different_context context each of which is bind to a normal distribution.
        Every normal distribution have a  fixed_variance (default = 0.1)
        Every normal distribution start with a mu in the range 1 to number_of_different_context
        Every normal distribution is different
        time_perturbation_function is a function of (time,mu) which modify the mu according to the time t

        """
        self.number_of_different_context = number_of_different_context
        self.number_of_observations = number_of_observations
        self.fixed_variance = fixed_variance
        self.time_perturbation_function = time_perturbation_function

        # Generate the contexts
        context_vectors, context_ids = make_blobs(
            n_samples=number_of_observations,
            n_features=3,
            centers=number_of_different_context,
            cluster_std=0.4,
            shuffle=True,
        )
        self.context_vectors = context_vectors
        self.context_ids = context_ids

        # Generate reward functions
        self.context_reward_parameters = {
            context_id: {"mu": context_id + 1, "sigma": fixed_variance}
            for context_id in range(number_of_different_context)
        }

    def generate_contexts(self) -> Iterable[np.ndarray]:
        for context_vector in self.context_vectors:
            yield context_vector

    def get_reward(self, action, context) -> bool:
        # Return true if the reward is positive, false otherwise

        mask_context = (self.context_vectors == context).all(1)
        assert sum(mask_context) == 1

        # Retreive the time step
        time = np.argwhere(mask_context).item()

        contex_group = self.context_ids[mask_context][0]
        context_reward_parameters = self.context_reward_parameters[contex_group]
        updated_mu = self.time_perturbation_function(
            time, context_reward_parameters["mu"]
        )
        success_prob = norm.cdf(
            action, loc=updated_mu, scale=context_reward_parameters["sigma"]
        )

        if np.random.rand() < success_prob:
            return True
        else:
            return False

    def get_best_reward_action(self, context):
        mask_context = (self.context_vectors == context).all(1)
        assert sum(mask_context) == 1

        # Retreive the time step
        time = np.argwhere(mask_context).item()

        contex_group = self.context_ids[mask_context][0]
        context_reward_parameters = self.context_reward_parameters[contex_group]
        updated_mu = self.time_perturbation_function(
            time, context_reward_parameters["mu"]
        )
        optimal_r = norm(loc=updated_mu, scale=context_reward_parameters["sigma"]).rvs()
        return optimal_r, optimal_r  # Not sure about this


if __name__ == "__main__":
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import numpy as np

    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=10_000,
        # time_perturbation_function=lambda time, mu: mu + (time // 100) * 5,
        time_perturbation_function=lambda time, mu: mu + np.cos(time/500),
    )
    bests_rewards = []
    for i, c in enumerate(environment.generate_contexts()):
        best_reward = environment.get_best_reward_action(c)
        bests_rewards.append(best_reward)

    plt.plot(np.arange(len(bests_rewards)),bests_rewards)
    plt.show()