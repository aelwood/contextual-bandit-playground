from __future__ import annotations

import abc
import collections
import copy
import math
import numpy as np
import typing
import scipy

import sklearn.linear_model as sklm
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

from scipy.stats import norm

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Iterable,
        NamedTuple,
        Optional,
        Sequence,
        Type,
        Union,
        Dict,
        Tuple,
    )


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

    @abc.abstractmethod
    def get_params(self):
        pass

    @abc.abstractmethod
    def __copy__(self):
        pass

    @abc.abstractmethod
    def __deepcopy__(self, memo):
        pass


class RandomPolicy(PolicyABC):
    def __init__(self, distribution):
        self.name = "random"
        assert type(distribution) == scipy.stats._distn_infrastructure.rv_frozen
        self.distribution = distribution

    def __copy__(self):
        return RandomPolicy(self.distribution)

    def __deepcopy__(self, memo):
        return RandomPolicy(copy.deepcopy(self.distribution, memo))

    def train(self):
        pass

    def notify_event(self, context, action, reward):
        pass

    def get_action(self, context):
        return self.distribution.rvs()

    def get_params(self):
        return None


class CyclicExploration(PolicyABC):
    def __init__(self, arm_values: Sequence[float]):
        self.name = "CyclicExploration"
        self.arm_values = arm_values
        self.last_called_item = 0

    def __copy__(self):
        return CyclicExploration(self.arm_values)

    def __deepcopy__(self, memo):
        return CyclicExploration(copy.deepcopy(self.arm_values, memo))

    def train(self):
        pass

    def notify_event(self, context, action, reward):
        pass

    def get_action(self, context):
        item_to_return = self.arm_values[self.last_called_item % len(self.arm_values)]
        self.last_called_item += 1
        return item_to_return

    def get_params(self):
        return None


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

    def get_params(self):
        return {
            "arm_values": self.arm_values,
            "epsilon": self.epsilon,
            "sw": self.sw,
        }

    def train(self):
        alphas = collections.defaultdict(int)
        betas = collections.defaultdict(int)
        arm_ids = []
        for action, reward in zip(
            self.past_actions[self.sw :], self.past_rewards[self.sw :]
        ):
            arm_id = self.values_arm[action]
            arm_ids.append(arm_id)
            is_success = reward
            if is_success:
                alphas[arm_id] += 1
            else:
                betas[arm_id] += 1

        for arm_id in np.unique(arm_ids):
            self.arms[arm_id] = scipy.stats.beta(
                a=alphas[arm_id] + 1, b=betas[arm_id] + 1
            )

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

    def __init__(self, arm_values: Dict[int, float], epsilon=0.02, sw=0):
        if sw == 0:
            self.name = "UCB1"
        else:
            self.name = f"UCB1_SW_{np.abs(sw)}"

        super(UcbPolicy, self).__init__(arm_values, epsilon, sw)

    def __copy__(self):
        return UcbPolicy(self.arm_values, self.epsilon, self.sw)

    def __deepcopy__(self, memo):
        return UcbPolicy(copy.deepcopy(self.arm_values, self.epsilon, self.sw, memo))

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
    def __init__(self, arm_values: Dict[int, float], epsilon=0.02, sw=0):
        self.name = ""
        if sw == 0:
            self.name = "TS"
        else:
            self.name = f"TS_SW_{np.abs(sw)}"

        super(ThompsonSamplingPolicy, self).__init__(arm_values, epsilon, sw)

    def __copy__(self):
        return ThompsonSamplingPolicy(self.arm_values, self.epsilon, self.sw)

    def __deepcopy__(self, memo):
        return ThompsonSamplingPolicy(
            copy.deepcopy(self.arm_values, self.epsilon, self.sw, memo)
        )

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
        p = np.dot(self.theta.T, x) + self.alpha * np.sqrt(
            np.dot(x.T, np.dot(A_inv, x))
        )

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
        self.name = f"LINUCB_a_{str(alpha).replace('.','')}"
        self.arm_values = arm_values
        self.values_arm = {v: k for k, v in arm_values.items()}
        self.alpha = alpha
        self.n_contex_features = n_contex_features

        self.arms = {
            a_id: LinUcbDisjointArm(arm_index=a_id, d=n_contex_features, alpha=alpha)
            for a_id in self.arm_values.keys()
        }

        sw *= -1 if sw > 0 else 1
        self.sw = sw

        self.past_contexts = []
        self.past_rewards = []
        self.past_actions = []

    def __copy__(self):
        return LinUcbPolicy(
            self.arm_values, self.n_contex_features, self.alpha, self.sw
        )

    def __deepcopy__(self, memo):
        return LinUcbPolicy(
            copy.deepcopy(
                self.arm_values, self.n_contex_features, self.alpha, self.sw, memo
            )
        )

    def get_params(self):
        return {
            "arm_values": self.arm_values,
            "alpha": self.alpha,
            "sw": self.sw,
        }

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

        for action, reward, context in zip(
            self.past_actions[self.sw :],
            self.past_rewards[self.sw :],
            self.past_contexts[self.sw :],
        ):
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
    def train(
        self,
        past_contexts: Sequence[np.ndarray],
        past_rewards: Sequence[float],
        past_actions: Sequence[float],
    ):
        pass

    @abc.abstractmethod
    def predict_reward(self, action, context: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def predict_reward_maintaining_graph(self, action, context: np.ndarray) -> float:
        pass

    @property
    def name(self):
        return self.__class__


class RewardEstimatorWithDataPreparationABC(RewardEstimatorABC, metaclass=abc.ABCMeta):
    def _prepare_x(self, contexts: Sequence[np.ndarray], actions: Sequence[float]):
        # TODO implement robust normalisation of data here?
        return np.concatenate(
            [np.array(contexts), np.array(actions).reshape(-1, 1)], axis=1
        )

    def _prepare_y(self, y):
        return np.array(y)


class LinearRegressionEstimatorABC(RewardEstimatorWithDataPreparationABC):
    @property
    def name(self):
        return self.__class__

    @abc.abstractmethod
    def get_model(self):
        pass

    def train(
        self,
        past_contexts: Sequence[np.ndarray],
        past_rewards: Sequence[float],
        past_actions: Sequence[float],
    ):
        X = self._prepare_x(past_contexts, past_actions)
        y = self._prepare_y(past_rewards)
        self.model = self.get_model()
        self.model.fit(X, y)

    def predict_reward(self, action, context: np.ndarray) -> float:
        assert (
            self.model is not None
        ), "Model likely hasn't been trained before being used"
        X = self._prepare_x([context], [action])
        r = self.model.predict(X)
        return r[0]

    def predict_reward_maintaining_graph(self, action, context: np.ndarray) -> float:
        coef = self.model.coef_
        intercept = self.model.intercept_

        return (
            tf.tensordot(
                tf.convert_to_tensor(coef, dtype="float32"),
                tf.concat([context, tf.expand_dims(action, 0)], 0),
                1,
            )
            + intercept
        )


class RidgeRegressionEstimator(LinearRegressionEstimatorABC):
    def __init__(self, alpha_l2=1.0):
        self.model: Optional[sklm.Ridge] = None  # DICT OF MODELS
        self.alpha_l2 = alpha_l2

    def get_model(self):
        return sklm.Ridge(alpha=self.alpha_l2)


class LogisticRegressionEstimator(LinearRegressionEstimatorABC):
    def __init__(self):
        self.model: Optional[sklm.LogisticRegression] = None  # DICT OF MODELS

    def get_model(self):
        return sklm.LogisticRegression()

    def _prepare_y(self, y):
        return np.array(y)


class RidgeRegressionEstimatorModelPerArm(RidgeRegressionEstimator):
    def __init__(self, actions, **kwargs):
        self.actions = actions
        self.model: Optional[Dict[int, sklm.Ridge]] = None
        super(RidgeRegressionEstimatorModelPerArm, self).__init__(**kwargs)

    def train(
        self,
        past_contexts: Sequence[np.ndarray],
        past_rewards: Sequence[float],
        past_actions: Sequence[float],
    ):
        model_dict = {}
        for action in self.actions:
            X = np.array(past_contexts)[past_actions == action]
            y = np.array(past_rewards)[past_actions == action]
            model = sklm.Ridge(alpha=self.alpha_l2)
            model.fit(X, y)
            model_dict[action] = model

        self.model = model_dict

    def predict_reward(self, action, context: np.ndarray) -> float:
        assert (
            self.model[action] is not None
        ), "Model likely hasn't been trained before being used"
        X = np.array(context).reshape(1, -1)
        r = self.model[action].predict(X)
        return r[0]

    def predict_reward_maintaining_graph(self, action, context: np.ndarray) -> float:
        # LINEARM MODELS = # ACTION

        coef = self.model[action].coef_
        intercept = self.model[action].intercept_

        return (
            tf.tensordot(
                tf.convert_to_tensor(coef, dtype="float32"),
                context,
                1,
            )
            + intercept
        )


class RewardLimiterMixin:
    def __init__(
        self,
        *,
        action_bounds: Tuple[float, float],
        reward_bounds: Tuple[Optional[float], Optional[float]],
        force_negative: bool = False,
        **kwargs,
    ):
        self.action_bounds = action_bounds
        self.reward_bounds = reward_bounds
        self.force_negative = force_negative
        if force_negative:
            assert self.reward_bounds[1] is not None

        super(RewardLimiterMixin, self).__init__(**kwargs)

    def predict_reward(self, action, context: np.ndarray) -> float:
        if not self.action_bounds[0] <= action <= self.action_bounds[1]:
            r = 0  # TODO: do we want 0 here or -inf? 0 makes more sense to me...
        else:
            r = super(RewardLimiterMixin, self).predict_reward(action, context)
            if self.reward_bounds[1] is not None:
                r = min(r, self.reward_bounds[1])
            if self.reward_bounds[0] is not None:
                r = max(r, self.reward_bounds[0])

        if self.force_negative:
            return r - self.reward_bounds[1]  # + self.reward_bounds[0]
        else:
            return r

    def predict_reward_maintaining_graph(self, action, context: np.ndarray) -> float:
        if self.force_negative:
            raise NotImplementedError

        # # clip reward to 0
        # return (
        #     tf.sigmoid(-(action - self.action_bounds[1]) * 1000)
        #     * tf.sigmoid((action - self.action_bounds[0]) * 1000)
        #     * tf.math.maximum(
        #         self.reward_bounds[0],
        #         tf.math.minimum(
        #             self.reward_bounds[1],
        #             super(RewardLimiterMixin, self).predict_reward_maintaining_graph(
        #                 action, context
        #             ),
        #         ),
        #     )
        # )

        # clip reward to - 999999
        return (
            (tf.sigmoid(float(action - self.action_bounds[1]) * 1000.0) * -999999 + 1)
            + (
                tf.sigmoid(-1.0 * (action - self.action_bounds[0]) * 1000.0) * -999999
                + 1
            )
            + tf.math.maximum(
                float(self.reward_bounds[0]),
                tf.math.minimum(
                    float(self.reward_bounds[1]),
                    super(RewardLimiterMixin, self).predict_reward_maintaining_graph(
                        action, context
                    ),
                ),
            )
        )


class LimitedRidgeRegressionEstimator(RewardLimiterMixin, RidgeRegressionEstimator):
    pass


class LimitedLogisticRegressionEstimator(
    RewardLimiterMixin, LogisticRegressionEstimator
):
    pass


class LimitedRidgeRegressionEstimatorModelPerArm(
    RewardLimiterMixin, RidgeRegressionEstimatorModelPerArm
):
    pass


class NeuralNetworkRewardEstimator(RewardEstimatorWithDataPreparationABC):
    """This is a class that predicts a reward given a context"""

    def __init__(
        self,
        layers: Sequence[int],
        context_vector_size: int,
        sigmoid_on_output: bool = True,
        epochs: int = 10,
        batch_size: int = 2,
    ):
        self.layers = layers
        self.context_vector_size = context_vector_size
        self.sigmoid_on_output = sigmoid_on_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._get_model()

    def _get_model(self):
        inputs = keras.Input(shape=(self.context_vector_size + 1,))
        x = inputs
        for l in self.layers:
            x = layers.Dense(l, activation="relu")(x)
        if self.sigmoid_on_output:
            output = layers.Dense(1, activation="sigmoid")(x)
        else:
            output = layers.Dense(1, activation=None)(x)

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def train(
        self,
        past_contexts: Sequence[np.ndarray],
        past_rewards: Sequence[float],
        past_actions: Sequence[float],
    ):

        X = self._prepare_x(past_contexts, past_actions)
        y = self._prepare_y(past_rewards)
        history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            sample_weight=self.get_training_weights(past_actions),
        )

    def get_training_weights(self, past_actions):
        return None

    def predict_reward(self, action, context: np.ndarray) -> float:
        X = self._prepare_x([context], [action])
        r = self.model.predict(X)
        return r[0][0]

    def predict_reward_maintaining_graph(self, action, context: np.ndarray) -> float:

        input_data = tf.concat([context, tf.expand_dims(action, 0)], 0)
        input_data = tf.reshape(input_data, shape=(1, -1))
        processed_data = self.model(input_data)

        return tf.squeeze(processed_data)


class OverrideGetTrainingWeightsSigmoid:
    def get_training_weights(self, past_actions):
        short_term_mem = 200
        long_term_mem = 100
        short_term_mem = min(short_term_mem, len(past_actions))
        long_term_mem = max(min(long_term_mem, len(past_actions) - short_term_mem), 0)

        memory = short_term_mem + long_term_mem

        starting_point = len(past_actions) - memory
        if long_term_mem > 0:
            q = 1 / long_term_mem
            a = len(past_actions) - long_term_mem - short_term_mem
            b = len(past_actions) - short_term_mem
            custom_sigmoid = lambda x: 1 / (
                1 + np.exp((q * 4) * (-x + a + (b - a) / 2))
            )
            weights = custom_sigmoid(np.arange(1, len(past_actions) + 1))
        else:
            weights = np.array([1.0] * len(past_actions))
        return weights


class LimitedNeuralNetworkRewardEstimator(
    RewardLimiterMixin, NeuralNetworkRewardEstimator
):
    pass


class LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
    RewardLimiterMixin, OverrideGetTrainingWeightsSigmoid, NeuralNetworkRewardEstimator
):
    pass


class MaxEntropyModelFreeABC(PolicyABC, metaclass=abc.ABCMeta):
    def __init__(
        self,
        reward_estimator: RewardEstimatorABC,
        alpha_entropy: float,
        pretrain_time: int,
        pretrain_policy: PolicyABC,
    ):
        self.alpha_entropy = alpha_entropy
        self.reward_estimator = reward_estimator

        self.past_contexts = []
        self.past_rewards = []
        self.past_actions = []

        self.pretrain_time = pretrain_time
        self.pretrain_policy = pretrain_policy
        self.pretrain_counter = 0

    @abc.abstractmethod
    def __copy__(self):
        pass

    @abc.abstractmethod
    def __deepcopy__(self, memo):
        pass

    def train(self):
        if self.pretrain_counter < self.pretrain_time:
            self.pretrain_policy.train()
        else:  # TODO: Ask Adam if it is ok for him
            self.reward_estimator.train(
                self.past_contexts, self.past_rewards, self.past_actions
            )

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

    def get_params(self):
        return {
            "alpha_entropy": self.alpha_entropy,
            "reward_estimator": self.reward_estimator.name,
        }


class MaxEntropyModelFreeDiscrete(MaxEntropyModelFreeABC):
    """This is equivalent to the method described in  Contextual bandit Shannon Entropy exploration:
    http://ras.papercept.net/images/temp/IROS/files/1465.pdf"""

    def __init__(
        self,
        *,
        possible_actions: Sequence[float],
        name="MaxEntropyModelFreeDiscrete",
        **kwargs,
    ):
        self.name = name
        self.possible_actions = possible_actions
        super(MaxEntropyModelFreeDiscrete, self).__init__(**kwargs)

    def __copy__(self):
        return MaxEntropyModelFreeDiscrete(
            **{
                "possible_actions": self.possible_actions,
                "name": self.name,
                "reward_estimator": self.reward_estimator,
                "alpha_entropy": self.alpha_entropy,
                "pretrain_time": self.pretrain_time,
                "pretrain_policy": self.pretrain_policy,
            }
        )

    def __deepcopy__(self, memo):
        return MaxEntropyModelFreeDiscrete(
            copy.deepcopy(
                self.possible_actions,
                self.name,
                self.reward_estimator,
                self.alpha_entropy,
                self.pretrain_time,
                self.pretrain_policy,
                memo,
            )
        )

    def get_params(self):
        fd = super(MaxEntropyModelFreeDiscrete, self).get_params()
        fd["possible_actions"] = self.possible_actions
        return fd

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
    def __init__(
        self,
        *,
        mcmc_initial_state: float,
        name="MaxEntropyModelFreeContinuous",
        **kwargs,
    ):
        self.mcmc_initial_state = mcmc_initial_state
        self.name = name
        super(MaxEntropyModelFreeContinuousABC, self).__init__(**kwargs)

    def get_params(self):
        fd = super(MaxEntropyModelFreeContinuousABC, self).get_params()
        fd["mcmc_initial_state"] = self.mcmc_initial_state
        return fd

    @abc.abstractmethod
    def _get_mcmc_kernel(self, log_prob_function):
        pass

    def _get_action_after_pretrain(self, context: np.ndarray) -> float:
        # Here we need to implement sampling from an MCMC type thing

        r = self.reward_estimator.predict_reward_maintaining_graph

        alpha = self.alpha_entropy

        def unnormalized_log_prob(a):
            return r(a, context) / alpha

        state = tfp.mcmc.sample_chain(
            num_results=1,
            num_burnin_steps=100,
            current_state=self.mcmc_initial_state,
            kernel=self._get_mcmc_kernel(log_prob_function=unnormalized_log_prob),
            trace_fn=None,
        )

        return float(state)


class MaxEntropyModelFreeContinuousHmc(MaxEntropyModelFreeContinuousABC):
    def __init__(self, **kwargs):
        self.name = "MaxEntropyModelFreeContinuousHmc"
        super(MaxEntropyModelFreeContinuousHmc, self).__init__(**kwargs)

    def __copy__(self):
        return MaxEntropyModelFreeContinuousHmc(
            **{
                "mcmc_initial_state": self.mcmc_initial_state,
                "reward_estimator": self.reward_estimator,
                "alpha_entropy": self.alpha_entropy,
                "pretrain_time": self.pretrain_time,
                "pretrain_policy": self.pretrain_policy,
                "name": self.name,
            }
        )

    def __deepcopy__(self, memo):
        return MaxEntropyModelFreeContinuousHmc(
            copy.deepcopy(
                self.mcmc_initial_state,
                self.reward_estimator,
                self.alpha_entropy,
                self.pretrain_time,
                self.pretrain_policy,
                self.name,
                memo,
            )
        )

    def _get_mcmc_kernel(self, log_prob_function):
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_function, step_size=1.0, num_leapfrog_steps=2
        )
