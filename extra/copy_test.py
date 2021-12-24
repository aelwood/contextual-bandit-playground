import numpy as np

from environments import SyntheticEnvironment
from evaluator import Evaluator
from policies import (
    RandomPolicy,
    UcbPolicy,
    LinUcbPolicy,
    MaxEntropyModelFreeDiscrete,
    RidgeRegressionEstimator,
    ThompsonSamplingPolicy,
    MaxEntropyModelFreeContinuousHmc,
    LimitedRidgeRegressionEstimator,
    LimitedRidgeRegressionEstimatorModelPerArm,
    CyclicExploration,
    LimitedLogisticRegressionEstimator,
    LimitedNeuralNetworkRewardEstimator,
    LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid,
)


import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from scipy.stats import uniform

possible_policies = [
    # UcbPolicy({k: v for k, v in enumerate(default_actions_range)}),
    # UcbPolicy({k: v for k, v in enumerate(default_actions_range)}, sw=-200),
    # ThompsonSamplingPolicy({k: v for k, v in enumerate(default_actions_range)}),
    # ThompsonSamplingPolicy(
    #     {k: v for k, v in enumerate(default_actions_range)}, sw=-200
    # ),
]

default_actions_range = np.arange(1, 10, 1)
default_steps_before_retraining_nn = 10
# MEMFD_MPA_A002
possible_policies.append(
    MaxEntropyModelFreeDiscrete(
        possible_actions=default_actions_range,
        name=f'MEMFD_MPA_a{str(0.02).replace(".", "")}',
        alpha_entropy=0.02,
        reward_estimator=LimitedRidgeRegressionEstimatorModelPerArm(
            actions=default_actions_range,
            alpha_l2=1.0,
            action_bounds=[
                default_actions_range[0],
                default_actions_range[-1],
            ],
            reward_bounds=[0, 1],
        ),
        pretrain_time=10,
        pretrain_policy=CyclicExploration(default_actions_range),
    )
)

# MEMFD_MPA_A01
possible_policies.append(
    MaxEntropyModelFreeDiscrete(
        possible_actions=default_actions_range,
        name=f'MEMFD_MPA_a{str(0.1).replace(".", "")}',
        alpha_entropy=0.1,
        reward_estimator=LimitedRidgeRegressionEstimatorModelPerArm(
            actions=default_actions_range,
            alpha_l2=1.0,
            action_bounds=[
                default_actions_range[0],
                default_actions_range[-1],
            ],
            reward_bounds=[0, 1],
        ),
        pretrain_time=10,
        pretrain_policy=CyclicExploration(default_actions_range),
    )
)

# MEMF_NN_10_10_A002
nn_layers = [10, 10]
alpha = 0.02
possible_policies.append(
    MaxEntropyModelFreeDiscrete(
        possible_actions=default_actions_range,
        name=f'MEMFD_NN_{[str(x) + "_" for x in nn_layers]}_a{str(alpha).replace(".", "")}',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimator(
            action_bounds=[
                default_actions_range[0],
                default_actions_range[-1],
            ],
            reward_bounds=(0.0, 1.0),
            layers=nn_layers,
            context_vector_size=3,
        ),
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )
)

# MEMF_NN_10_10_A005
nn_layers = [10, 10]
alpha = 0.05
possible_policies.append(
    MaxEntropyModelFreeDiscrete(
        possible_actions=default_actions_range,
        name=f'MEMFD_NN_{[str(x) + "_" for x in nn_layers]}_a{str(alpha).replace(".", "")}',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimator(
            action_bounds=[
                default_actions_range[0],
                default_actions_range[-1],
            ],
            reward_bounds=(0.0, 1.0),
            layers=nn_layers,
            context_vector_size=3,
        ),
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )
)

# MEMF_NN_50_50_A01
nn_layers = [50, 50]
alpha = 0.1
possible_policies.append(
    MaxEntropyModelFreeDiscrete(
        possible_actions=default_actions_range,
        name=f'MEMFD_NN_{[str(x) + "_" for x in nn_layers]}_a{str(alpha).replace(".", "")}',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimator(
            action_bounds=[
                default_actions_range[0],
                default_actions_range[-1],
            ],
            reward_bounds=(0.0, 1.0),
            layers=nn_layers,
            context_vector_size=3,
        ),
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )
)

# MEMF_HMC_NN_50_50_A005
nn_layers = [50, 50]
alpha = 0.05
possible_policies.append(
    MaxEntropyModelFreeContinuousHmc(
        mcmc_initial_state=5.0,
        name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimator(
            # action_bounds=[default_actions_range[0], default_actions_range[-1]],
            action_bounds=[1.0, 10.0],
            reward_bounds=(0.0, 1.0),
            layers=nn_layers,
            context_vector_size=3,
        ),
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )
)

# MEMF_HMC_NN_10_A01
nn_layers = [10]
alpha = 0.1
possible_policies.append(
    MaxEntropyModelFreeContinuousHmc(
        mcmc_initial_state=5.0,
        name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimator(
            # action_bounds=[default_actions_range[0], default_actions_range[-1]],
            action_bounds=[1.0, 10.0],
            reward_bounds=(0.0, 1.0),
            layers=nn_layers,
            context_vector_size=3,
        ),
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )
)

#####

for policy_base in possible_policies:
    ciao = policy_base.__copy__()
    print(f"{ciao.name} copied!")

1 == 1
