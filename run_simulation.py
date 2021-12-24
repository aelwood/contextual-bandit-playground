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


def simulate(
    environment, policy, evaluator, evaluation_frequency=100, steps_to_train=1
):
    evaluation_frequency = evaluation_frequency
    for i, c in enumerate(environment.generate_contexts()):
        a = policy.get_action(c)
        r, s_r = environment.get_reward(a, c)
        policy.notify_event(c, a, s_r)

        optimal_r, optimal_a, stochastic_r = environment.get_best_reward_action(c)

        evaluator.notify(a, r, s_r, optimal_a, optimal_r, stochastic_r)
        if i % steps_to_train == 0:
            policy.train()

        if i % evaluation_frequency == 0 and i > 0:
            print(evaluator.get_stats())

    print("Final results")
    print(evaluator.get_stats())
    evaluator.end()


def tes_MaxEntropyModelFreeContinuousHmc():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 200) + 0.5,
        name="2c_dm_fst",
    )

    nn_layers = [50]
    alpha = 0.05

    policy = MaxEntropyModelFreeContinuousHmc(
        mcmc_initial_state=5.0,
        name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}_SIGMOID_WEIGHT',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
            action_bounds=[1.0, 10.0],
            reward_bounds=(0.0, 1.0),
            layers=nn_layers,
            context_vector_size=3,
        ),
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )

    evaluator = Evaluator(
        run_name=f"{policy.name}_TEST0",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
        experiment_name=environment.name,
    )
    simulate(
        environment, policy, evaluator, evaluation_frequency=100, steps_to_train=10
    )


if __name__ == "__main__":
    run_ablation_test = False

    if not run_ablation_test:
        tes_MaxEntropyModelFreeContinuousHmc()
    else:
        number_of_observations = 150  # TODO: 2_000
        possible_environments = [
            SyntheticEnvironment(
                number_of_different_context=1,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu,
                name="1c_st",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu,
                name="2c_st",
            ),
            # SyntheticEnvironment(
            #     number_of_different_context=1,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu,
            #     environment_best_action_offset=0.2,
            #     name="1c_st_offset",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=2,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu,
            #     environment_best_action_offset=0.2,
            #     name="2c_st_offset",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=1,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu
            #     + np.cos(time / 1_000)
            #     + 0.5,
            #     name="1c_dm_slw",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=2,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu
            #     + np.cos(time / 1_000)
            #     + 0.5,
            #     name="2c_dm_slw",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=1,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu
            #     + np.cos(time / 500)
            #     + 0.5,
            #     name="1c_dm",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=2,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu
            #     + np.cos(time / 500)
            #     + 0.5,
            #     name="2c_dm",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=1,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu
            #     + np.cos(time / 200)
            #     + 0.5,
            #     name="1c_dm_fst",
            # ),
            # SyntheticEnvironment(
            #     number_of_different_context=2,
            #     number_of_observations=number_of_observations,
            #     time_perturbation_function=lambda time, mu: mu
            #     + np.cos(time / 200)
            #     + 0.5,
            #     name="2c_dm_fst",
            # ),
        ]

        default_actions_range = np.arange(1, 10, 1)
        default_steps_before_retraining_nn = 10

        possible_policies = [
            # UcbPolicy({k: v for k, v in enumerate(default_actions_range)}),
            # UcbPolicy({k: v for k, v in enumerate(default_actions_range)}, sw=-200),
            # ThompsonSamplingPolicy({k: v for k, v in enumerate(default_actions_range)}),
            # ThompsonSamplingPolicy(
            #     {k: v for k, v in enumerate(default_actions_range)}, sw=-200
            # ),
        ]

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
            for environment in possible_environments:
                policy = policy_base.__copy__()
                print(f"Running {policy.name} - {environment.name}")

                evaluator = Evaluator(
                    run_name=f"{policy.name}",
                    save_data=True,
                    plot_data=False,
                    use_mlflow=True,
                    policy=policy,
                    environment=environment,
                    experiment_name=environment.name,
                )

                steps_to_train = 1
                if "NN" in policy.name:
                    steps_to_train = default_steps_before_retraining_nn

                simulate(
                    environment,
                    policy,
                    evaluator,
                    evaluation_frequency=100,
                    steps_to_train=steps_to_train,
                )
