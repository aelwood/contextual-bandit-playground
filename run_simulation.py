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


def single_context_static_reward_random_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=200,
        time_perturbation_function=lambda time, mu: mu,
    )

    policy = RandomPolicy(uniform(loc=0.5, scale=10))
    evaluator = Evaluator(
        run_name="single_context_static_reward_random_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_cyclicexploration_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=100,
        time_perturbation_function=lambda time, mu: mu,
    )
    policy = CyclicExploration(np.arange(1, 6, 2))

    evaluator = Evaluator(
        run_name="single_context_static_reward_cyclicexploration_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_ucb_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(1, 6, 2))})

    evaluator = Evaluator(
        run_name="single_context_static_reward_ucb_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_ThompsonSampling_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
    )
    policy = ThompsonSamplingPolicy({k: v for k, v in enumerate(np.arange(1, 6, 2))})
    evaluator = Evaluator(
        run_name="single_context_static_reward_ThompsonSampling_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_dynamic_reward_ucb_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 1_000) + 0.5,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0.0, 3, 0.5))})
    evaluator = Evaluator(
        run_name="single_context_dynamic_reward_ucb_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_dynamic_reward_ThompsonSampling_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = ThompsonSamplingPolicy(
        {k: v for k, v in enumerate(np.arange(0.0, 3, 0.5))}
    )
    evaluator = Evaluator(
        run_name="single_context_dynamic_reward_ThompsonSampling_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_dynamic_reward_ThompsonSampling_SW_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = ThompsonSamplingPolicy(
        {k: v for k, v in enumerate(np.arange(0.0, 3, 0.5))}, sw=-200
    )
    evaluator = Evaluator(
        run_name="single_context_dynamic_reward_ThompsonSampling_SW_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_dynamic_reward_ucb_sw_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0.0, 3, 0.5))}, sw=-200)
    evaluator = Evaluator(
        run_name="single_context_dynamic_reward_ucb_sw_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def duble_context_dynamic_reward_ucb_sw_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0.0, 7, 1.0))}, sw=-200)
    evaluator = Evaluator(
        run_name="duble_context_dynamic_reward_ucb_sw_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def duouble_context_static_reward_ucb_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0.0, 7, 1.0))})
    evaluator = Evaluator(
        run_name="duouble_context_static_reward_ucb_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_LinUcbPolicy_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=1_000,
        time_perturbation_function=lambda time, mu: mu,
        n_context_features=2,
    )
    policy = LinUcbPolicy({k: v for k, v in enumerate(np.arange(1, 10, 4))}, 2, 0.0)
    evaluator = Evaluator(
        run_name="single_context_static_reward_LinUcbPolicy_policy",
        save_data=False,
        plot_data=True,
        use_mlflow=False,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def double_context_static_reward_LinUcbPolicy_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
        n_context_features=2,
    )
    policy = LinUcbPolicy({k: v for k, v in enumerate(np.arange(1, 7, 2))}, 2, 0.01)
    evaluator = Evaluator(
        run_name="double_context_static_reward_LinUcbPolicy_policy",
        save_data=False,
        plot_data=True,
        use_mlflow=False,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def double_context_dynamic_reward_LinUcbPolicy_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
        n_context_features=2,
    )
    policy = LinUcbPolicy({k: v for k, v in enumerate(np.arange(1, 7, 0.5))}, 2, 0.01)
    evaluator = Evaluator(
        run_name="double_context_dynamic_reward_LinUcbPolicy_policy",
        save_data=True,
        plot_data=False,
        use_mlflow=True,
        policy=policy,
        environment=environment,
    )

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_hmc_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=200,
        time_perturbation_function=lambda time, mu: mu,
    )

    # reward_estimator = RidgeRegressionEstimator(alpha_l2=1.0)
    reward_estimator = LimitedRidgeRegressionEstimator(
        alpha_l2=1.0,
        action_bounds=(0.0, 10.0),
        reward_bounds=(0.0, 1.0),
        force_negative=False,
    )
    pretrain_policy = RandomPolicy(uniform(loc=0.5, scale=10))
    policy = MaxEntropyModelFreeContinuousHmc(
        mcmc_initial_state=0.5,
        alpha_entropy=0.02,
        reward_estimator=reward_estimator,
        pretrain_time=10,
        pretrain_policy=pretrain_policy,
    )
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_hmc_policy_nn():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=200,
        time_perturbation_function=lambda time, mu: mu,
    )

    # reward_estimator = RidgeRegressionEstimator(alpha_l2=1.0)
    reward_estimator = LimitedNeuralNetworkRewardEstimator(
        action_bounds=(0.0, 10.0),
        reward_bounds=(0.0, 1.0),
        layers=[50, 50],
        context_vector_size=3,
    )
    pretrain_policy = RandomPolicy(uniform(loc=0.5, scale=10))
    policy = MaxEntropyModelFreeContinuousHmc(
        mcmc_initial_state=0.5,
        alpha_entropy=0.1,
        reward_estimator=reward_estimator,
        pretrain_time=10,
        pretrain_policy=pretrain_policy,
    )
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def single_context_static_reward_model_free_discrete_policy_nn():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=200,
        time_perturbation_function=lambda time, mu: mu,
    )

    # reward_estimator = RidgeRegressionEstimator(alpha_l2=1.0)
    reward_estimator = LimitedNeuralNetworkRewardEstimator(
        action_bounds=(0.0, 10.0),
        reward_bounds=(0.0, 1.0),
        layers=[50, 50],
        context_vector_size=3,
    )
    policy = MaxEntropyModelFreeDiscrete(
        possible_actions=np.arange(0, 10, 1),
        alpha_entropy=0.02,
        reward_estimator=reward_estimator,
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )

    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)


def double_context_static_reward_model_free_discrete_policy_nn():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=200,
        time_perturbation_function=lambda time, mu: mu,
    )

    # reward_estimator = RidgeRegressionEstimator(alpha_l2=1.0)
    reward_estimator = LimitedNeuralNetworkRewardEstimator(
        action_bounds=(0.0, 10.0),
        reward_bounds=(0.0, 1.0),
        layers=[5, 5],
        context_vector_size=3,
    )
    policy = MaxEntropyModelFreeDiscrete(
        possible_actions=np.arange(0, 10, 1),
        alpha_entropy=0.02,
        reward_estimator=reward_estimator,
        pretrain_time=10,
        pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    )

    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)


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
        # single_context_static_reward_random_policy()
        # single_context_static_reward_cyclicexploration_policy()
        # single_context_static_reward_ucb_policy()
        # single_context_dynamic_reward_ucb_policy()
        # single_context_dynamic_reward_ucb_sw_policy() # ??? why
        # duouble_context_static_reward_ucb_policy()
        # duble_context_dynamic_reward_ucb_sw_policy()
        # single_context_static_reward_LinUcbPolicy_policy()
        # double_context_static_reward_LinUcbPolicy_policy()
        # single_context_static_reward_hmc_policy()
        # single_context_static_reward_hmc_policy_nn()
        # single_context_static_reward_model_free_discrete_policy_nn()
        # double_context_static_reward_model_free_discrete_policy_nn()
        tes_MaxEntropyModelFreeContinuousHmc()
    else:
        number_of_observations = 2_000
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
            SyntheticEnvironment(
                number_of_different_context=1,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu,
                environment_best_action_offset=0.2,
                name="1c_st_offset",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu,
                environment_best_action_offset=0.2,
                name="2c_st_offset",
            ),
            SyntheticEnvironment(
                number_of_different_context=1,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 1_000)
                + 0.5,
                name="1c_dm_slw",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 1_000)
                + 0.5,
                name="2c_dm_slw",
            ),
            SyntheticEnvironment(
                number_of_different_context=1,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 500)
                + 0.5,
                name="1c_dm",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 500)
                + 0.5,
                name="2c_dm",
            ),
            SyntheticEnvironment(
                number_of_different_context=1,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 200)
                + 0.5,
                name="1c_dm_fst",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 200)
                + 0.5,
                name="2c_dm_fst",
            ),
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

        for alpha in [0.2, 0.1, 0.05, 0.02]:
            possible_policies.append(
                LinUcbPolicy(
                    {k: v for k, v in enumerate(default_actions_range)}, 3, alpha
                ),
            )
            possible_policies.append(
                MaxEntropyModelFreeDiscrete(
                    possible_actions=default_actions_range,
                    name=f'MEMFD_a{str(alpha).replace(".","")}',
                    alpha_entropy=alpha,
                    reward_estimator=LimitedRidgeRegressionEstimator(
                        alpha_l2=1.0,
                        action_bounds=[
                            default_actions_range[0],
                            default_actions_range[-1],
                        ],
                        reward_bounds=[0, 1],
                    ),
                    pretrain_time=10,
                    pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
                )
            )

            possible_policies.append(
                MaxEntropyModelFreeDiscrete(
                    possible_actions=default_actions_range,
                    name=f'MEMFD_MPA_a{str(alpha).replace(".","")}',
                    alpha_entropy=alpha,
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
            for nn_layers in [[10], [10, 10], [50], [50, 50]]:
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

        # possible_policies = []
        # for alpha in [0.2, 0.1, 0.05, 0.02]:
        #

        # MaxEntropyModelFreeDiscrete(
        #     possible_actions=default_actions_range,
        #     name='MaxEntropyModelFreeDiscreteLOGISTIC',
        #     alpha_entropy=0.02,
        #     reward_estimator=LimitedLogisticRegressionEstimator(
        #         action_bounds=[0, 8], reward_bounds=[0, 1],
        #     ),
        #     pretrain_time=10,
        #     pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
        # ),
        # MaxEntropyModelFreeContinuousHmc(
        #     mcmc_initial_state=0.5,
        #     alpha_entropy=0.2,
        #     reward_estimator=LimitedRidgeRegressionEstimator(
        #     alpha_l2=1.0,
        #     action_bounds=(0.0, 10.0),
        #     reward_bounds=(0.0, 1.0),
        #     force_negative=False,
        # ),
        #     pretrain_time=100,
        #     pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
        # ),

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
