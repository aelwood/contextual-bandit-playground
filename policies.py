from __future__ import annotations

import abc
import collections
import math
import numpy as np
import typing
import scipy

import sklearn.linear_model as sklm
import tensorflow as tf
import tensorflow_probability as tfp

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


class RandomPolicy(PolicyABC):
    def __init__(self, distribution):
        assert type(distribution) == scipy.stats._distn_infrastructure.rv_frozen
        self.distribution = distribution

    def train(self):
        pass

    def notify_event(self, context, action, reward):
        pass

    def get_action(self, context):
        return self.distribution.rvs()


class MABPolicyABC(PolicyABC):
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
            self.arms[arm_id] = scipy.stats.beta(a=alphas[arm_id] + 1, b=betas[arm_id] + 1)

    def notify_event(self, context, action, stochastic_reward):
        self.past_rewards.append(stochastic_reward)
        self.past_actions.append(action)

    def _get_total_pulls(self, arm_beta):
        return arm_beta.kwds["a"] + arm_beta.kwds["b"]


class UcbPolicy(MABPolicyABC):
    """
    This is the implementation of the UCB1 policy, found on this website:
    https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-upper-confidence-bound-algorithm-4b84be516047

    For this solution, context is not taken into account
    arm_values = dict arm id(int) -> arm value (float)
    epsilon = scalar from which consider the reward a positive reward
    sw = sliding windows parameters: number of samples to consider during the training phase
    """

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
        for id_, arm_beta in self.arms.items():
            num_of_pulls_from_the_arm = self._get_total_pulls(arm_beta)
            arm_mean_reward = arm_beta.mean()

            arm_value = self.arm_values[id_]

            if num_of_pulls_from_the_arm == 2:
                return float(arm_value)
            upper_bound_value = arm_mean_reward + math.sqrt(
                (2 * math.log(total_pulls)) / float(num_of_pulls_from_the_arm)
            )
            samples[arm_value] = upper_bound_value
        markup_of_max = max(samples, key=samples.get)
        return float(markup_of_max)


class ThompsonSamplingPolicy(MABPolicyABC):
    def get_action(self, context):
        samples = {}
        for id_, arm_beta in self.arms.items():
            arm_value = self.arm_values[id_]
            samples[arm_value] = arm_beta.rvs()
        markup_of_max = max(samples, key=samples.get)
        return float(markup_of_max)


class LinUcbDisjointArm:
    """
    Code for linear UCB adapted from:
    https://github.com/kfoofw/bandit_simulations/blob/master/python/contextual_bandits/notebooks/LinUCB_disjoint.ipynb
    """

    def __init__(self, arm_index, d, alpha):
        # Track arm index
        self.arm_index = arm_index

        # Keep track of alpha
        self.alpha = alpha
        self.d = d
        self.init_arms()

    def init_arms(self):
        # A: (d x d) matrix = D_a.T * D_a + I_d.
        # The inverse of A is used in ridge regression
        self.A = np.identity(self.d)

        # b: (d x 1) corresponding response vector.
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([self.d, 1])

    def calc_UCB(self, context):
        # Find A inverse for ridge regression
        A_inv = np.linalg.inv(self.A)

        # Perform ridge regression to obtain estimate of covariate coefficients theta
        # theta is (d x 1) dimension vector
        self.theta = np.dot(A_inv, self.b)

        # Reshape covariates input into (d x 1) shape vector
        x = context.reshape([-1, 1])

        # Find ucb based on p formulation (mean + std_dev)
        # p is (1 x 1) dimension vector
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(np.dot(x.T, np.dot(A_inv, x)))

        return p

    def reward_update(self, reward, context):
        # Reshape covariates input into (d x 1) shape vector
        x = context.reshape([-1, 1])

        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)

        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x


class LinUcbPolicy(PolicyABC):
    def __init__(self, arm_values: Dict[int, float], n_contex_features, alpha, sw=0):
        self.arm_values = arm_values
        self.values_arm = {v: k for k, v in arm_values.items()}

        self.arms = {
            a_id: LinUcbDisjointArm(arm_index=a_id, d=n_contex_features, alpha=alpha) for a_id in self.arm_values.keys()
        }

        sw *= -1 if sw > 0 else 1
        self.sw = sw

        self.past_contexts = []
        self.past_rewards = []
        self.past_actions = []

    def get_action(self, context):
        # Initiate ucb to be 0
        highest_ucb = -1

        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []

        for a_id in range(len(self.arms)):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.arms[a_id].calc_UCB(context)

            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                # Set new max ucb
                highest_ucb = arm_ucb

                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [a_id]

            # If there is a tie, append to candidate_arms
            elif arm_ucb == highest_ucb:
                candidate_arms.append(a_id)

        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)
        arm_value = self.arm_values[chosen_arm]

        return arm_value

    def train(self):
        # First we have to re-initialize the arms
        for arm in self.arms.values():
            arm.init_arms()

        for action, reward, context in zip(self.past_actions[self.sw:], self.past_rewards[self.sw:], self.past_contexts[self.sw:]):
            arm_id = self.values_arm[action]

            is_success = reward
            if is_success:
                self.arms[arm_id].reward_update(reward, context)

    def notify_event(self, context, action, stochastic_reward):
        self.past_contexts.append(context)
        self.past_rewards.append(stochastic_reward)
        self.past_actions.append(action)


class RewardEstimatorABC(metaclass=abc.ABCMeta):
    """This is a class that predicts a reward given a context"""

    @abc.abstractmethod
    def train(self, past_contexts: Sequence[np.ndarray], past_rewards: Sequence[float], past_actions: Sequence[float]):
        pass

    @abc.abstractmethod
    def predict_reward(self, action, context: np.ndarray) -> float:
        pass


class RidgeRegressionEstimator(RewardEstimatorABC):

    def __init__(self, alpha_l2 = 1.0):
        self.model: Optional[sklm.Ridge] = None
        self.alpha_l2 = alpha_l2

    def _prepare_x(self, contexts: Sequence[np.ndarray], actions: Sequence[float]):
        # TODO implement robust normalisation of data here?
        return np.concatenate([np.array(contexts), np.array(actions).reshape(-1,1)], axis=1)

    def train(self, past_contexts: Sequence[np.ndarray], past_rewards: Sequence[float], past_actions: Sequence[float]):
        X = self._prepare_x(past_contexts, past_actions)
        y = np.array(past_rewards)
        self.model = sklm.Ridge(alpha=self.alpha_l2)
        self.model.fit(X,y)

    def predict_reward(self, action, context: np.ndarray) -> float:
        assert self.model is not None, "Model likely hasn't been trained before being used"
        X = self._prepare_x([context], [action])
        r = self.model.predict(X)
        return r[0]


class MaxEntropyModelFreeABC(PolicyABC, metaclass=abc.ABCMeta):
    def __init__(self, reward_estimator: RewardEstimatorABC, alpha_entropy: float, pretrain_time: int, pretrain_policy: PolicyABC):
        self.alpha_entropy = alpha_entropy
        self.reward_estimator = reward_estimator

        self.past_contexts = []
        self.past_rewards = []
        self.past_actions = []

        self.pretrain_time = pretrain_time
        self.pretrain_policy = pretrain_policy
        self.pretrain_counter = 0

    def train(self):
        if self.pretrain_counter < self.pretrain_time:
            self.pretrain_policy.train()

        self.reward_estimator.train(self.past_contexts, self.past_rewards, self.past_actions)

    def notify_event(self, context: np.ndarray, action: float, reward: float):
        if self.pretrain_counter < self.pretrain_time:
            self.pretrain_policy.notify_event(context, action, reward)

        self.past_contexts.append(context)
        self.past_rewards.append(reward)
        self.past_actions.append(action)

    def get_action(self, context: np.ndarray) -> float:
        if self.pretrain_counter < self.pretrain_time:
            self.pretrain_counter += 1
            return self.pretrain_policy.get_action(context)
        else:
            return self._get_action_after_pretrain(context)

    @abc.abstractmethod
    def _get_action_after_pretrain(self, context: np.ndarray) -> float:
        pass


class MaxEntropyModelFreeDiscrete(MaxEntropyModelFreeABC):
    """This is equivalent to the method described in  Contextual bandit Shannon Entropy exploration:
    http://ras.papercept.net/images/temp/IROS/files/1465.pdf"""

    def __init__(self, *, possible_actions: Sequence[float], **kwargs):
        self.possible_actions = possible_actions
        super(MaxEntropyModelFreeDiscrete, self).__init__(**kwargs)

    def _get_action_after_pretrain(self, context):

        probabilities = []
        for a in self.possible_actions:
            r = self.reward_estimator.predict_reward(a, context)
            unnormed_p = np.exp(r / self.alpha_entropy)
            probabilities.append(unnormed_p)

        normalisation_factor = np.sum(probabilities)
        probabilities = np.array(probabilities) / normalisation_factor
        return np.random.choice(self.possible_actions, p=probabilities)


class MaxEntropyModelFreeContinuousABC(MaxEntropyModelFreeABC, metaclass=abc.ABCMeta):
    def __init__(self, *, mcmc_initial_state: float, **kwargs):
        self.mcmc_initial_state = mcmc_initial_state
        super(MaxEntropyModelFreeContinuousABC, self).__init__(**kwargs)

    @abc.abstractmethod
    def _get_mcmc_kernel(self, log_prob_function):
        pass

    def _get_action_after_pretrain(self, context: np.ndarray) -> float:
        # Here we need to implement sampling from an MCMC type thing

        r = self.reward_estimator.predict_reward
        alpha = self.alpha_entropy

        def unnormalized_log_prob(a):
            return r(a, context)/alpha

        state = tfp.mcmc.sample_chain(
            num_results=1,
            num_burnin_steps=500,
            current_state=self.mcmc_initial_state,
            kernel=self._get_mcmc_kernel(log_prob_function=unnormalized_log_prob),
            trace_fn=None)

        #sample = tf.reduce_mean(states, axis=0)

        return state


class MaxEntropyModelFreeContinuousHmc(MaxEntropyModelFreeContinuousABC):
    def _get_mcmc_kernel(self, log_prob_function):
        return tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=log_prob_function,
                step_size=1.0,
                num_leapfrog_steps=2
        )
