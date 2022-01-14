from __future__ import annotations

import abc
import numpy as np
import typing

from abc import ABC
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll, make_circles
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

    @abc.abstractmethod
    def get_params(self):
        pass


class SyntheticEnvironment(EnvironmentABC, ABC):
    def __init__(
        self,
        number_of_different_context: int,
        number_of_observations: int,
        time_perturbation_function,
        fixed_variances=0.6,
        n_context_features=3,
        environment_best_action_offset=1,
        action_offset = 2,
        name="default",
    ):
        """
        It creates number_of_different_context context each of which is bind to a normal distribution.
        Every normal distribution have a  fixed_variance (default = 0.6)
        Every normal distribution start with a mu in the range 1 to number_of_different_context*4 with an offset of 4
        Every normal distribution is different
        time_perturbation_function is a function of (time,mu) which modify the mu according to the time t

        """
        self.number_of_different_context = number_of_different_context
        self.number_of_observations = number_of_observations
        self.time_perturbation_function = time_perturbation_function
        self.name = name

        # Generate the contexts
        context_vectors, context_ids = self._generate_context(
            number_of_observations, n_context_features, number_of_different_context
        )

        # Normalize vectors:
        context_vectors -= context_vectors.mean(0)
        context_vectors /= context_vectors.std(0)

        self.context_vectors = context_vectors
        self.context_ids = context_ids

        if type(fixed_variances) in [float, int]:
            fixed_variances = [fixed_variances] * number_of_different_context
        else:
            assert len(fixed_variances) == number_of_different_context, (
                "The number of given variances should be equal "
                "to the number of different context "
            )


        # Generate reward functions
        self.context_reward_parameters = {
            context_id: {
                "mu": context_id * action_offset + environment_best_action_offset,
                "sigma": fixed_variances[context_id],
            }
            for context_id in range(number_of_different_context)
        }

    def _generate_context(
        self, number_of_observations, n_context_features, number_of_different_context
    ):
        context_vectors, context_ids = make_blobs(
            n_samples=number_of_observations,
            n_features=n_context_features,
            centers=number_of_different_context,
            cluster_std=0.4,
            shuffle=True,
        )
        return context_vectors, context_ids

    def get_params(self):
        return {
            "n_context": self.number_of_different_context,
            "context_reward_parameters": self.context_reward_parameters,
        }

    def generate_contexts(self) -> Iterable[np.ndarray]:
        for context_vector in self.context_vectors:
            yield context_vector

    def get_reward(self, action, context) -> (float, bool):
        mask_context = (self.context_vectors == context).all(1)
        assert sum(mask_context) == 1

        # Retreive the time step
        time = np.argwhere(mask_context).item()

        contex_group = self.context_ids[mask_context][0]
        context_reward_parameters = self.context_reward_parameters[contex_group]
        updated_mu = self.time_perturbation_function(
            time, context_reward_parameters["mu"]
        )
        reward = norm.pdf(
            action, loc=updated_mu, scale=context_reward_parameters["sigma"]
        )
        reward = np.round(reward, 4)
        stochastic_reward = np.random.rand() < reward
        return reward, stochastic_reward

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

        optimal_a = updated_mu

        optimal_r = norm.pdf(
            optimal_a, loc=updated_mu, scale=context_reward_parameters["sigma"]
        )
        optimal_r = np.round(optimal_r, 4)
        stochastic_reward = np.random.rand() < optimal_r

        return optimal_r, optimal_a, stochastic_reward


# , ,
# TODO: Moons and circles
class MoonsContextsMixin:
    def _generate_context(
        self, number_of_observations, n_context_features, number_of_different_context
    ):
        assert n_context_features == 2, "Context features must be 2"
        assert (
            number_of_different_context == 2
        ), "Number of different contexts must be 2"
        context_vectors, context_ids = make_moons(
            n_samples=number_of_observations,
            shuffle=True,
        )
        return context_vectors, context_ids


class CirclesContextsMixin:
    def _generate_context(
        self, number_of_observations, n_context_features, number_of_different_context
    ):
        assert n_context_features == 2, "Context features must be 2"
        assert (
            number_of_different_context == 2
        ), "Number of different contexts must be 2"
        context_vectors, context_ids = make_circles(
            n_samples=number_of_observations,
            shuffle=True,
        )
        return context_vectors, context_ids


class MoonSyntheticEnvironment(MoonsContextsMixin, SyntheticEnvironment):
    pass


class CirclesSyntheticEnvironment(CirclesContextsMixin, SyntheticEnvironment):
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    environment = CirclesSyntheticEnvironment(
        number_of_different_context=2,
        n_context_features=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500)*100,
        fixed_variances=30,
        environment_best_action_offset=300,
        action_offset=400,
    )
    # Plotting the contex space
    # for unique_id in np.unique(environment.context_ids):
    #     points = environment.context_vectors[environment.context_ids == unique_id]
    #     plt.scatter(points[:, 0], points[:, 1], label=unique_id)
    #
    # plt.show()
    # 1 == 1

    # Plotting the actions
    bests_rewards = []
    bests_context = []

    for i, c in enumerate(environment.generate_contexts()):
        (
            best_reward,
            best_context,
            stochastic_reward,
        ) = environment.get_best_reward_action(c)
        bests_rewards.append(best_reward)
        bests_context.append(best_context)

    plt.plot(
        np.arange(len(bests_rewards)),
        bests_rewards,
        label="reward",
        linestyle="",
        marker="^",
    )
    plt.plot(
        np.arange(len(bests_context)),
        bests_context,
        label="context",
        linestyle="",
        marker="^",
    )
    plt.legend()
    plt.show()
