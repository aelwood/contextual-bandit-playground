import numpy as np

from environments import (
    SyntheticEnvironment,
    MoonSyntheticEnvironment,
    CirclesSyntheticEnvironment,
)
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
    environment = CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=30,
                time_perturbation_function=lambda time, mu: mu
                                                            + np.cos(time / 200)*100
                                                            + 0.5,
                fixed_variances=30,
                environment_best_action_offset=300,
                action_offset=400,
                name="2c_dm_fst_circ",
            )

    nn_layers = [50]
    alpha = 0.05

    # policy = MaxEntropyModelFreeContinuousHmc(
    #     mcmc_initial_state=(0,1_000),
    #     num_burnin_steps=50,
    #     name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}_SIGMOID_WEIGHT',
    #     alpha_entropy=alpha,
    #     reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
    #         action_bounds=[1.0, 10.0],
    #         reward_bounds=(0.0, 1.0),
    #         layers=nn_layers,
    #         context_vector_size=3,
    #     ),
    #     pretrain_time=10,
    #     pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
    # )

    mem = 200
    policy = MaxEntropyModelFreeContinuousHmc(
        mcmc_initial_state="last_state",
        num_burnin_steps=50,
        name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}_sig_{mem}',
        alpha_entropy=alpha,
        reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
            short_term_mem=mem,
            long_term_mem=mem,
            action_bounds=[1, 1_000],
            reward_bounds=reward_bounds,
            layers=nn_layers,
            context_vector_size=2,
        ),
        pretrain_time=pretrain_time,
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
    run_ablation_test = True

    if not run_ablation_test:
        tes_MaxEntropyModelFreeContinuousHmc()
    else:
        number_of_observations = 2_000
        possible_environments = [
            CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                                                            + np.cos(time / 1_000)*100
                                                            + 0.5,
                fixed_variances=30,
                environment_best_action_offset=300,
                action_offset=400,
                mul_factor=120,
                name="2c_dm_slw_circ",
            ),
            CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                                                            + np.cos(time / 500)*100
                                                            + 0.5,
                fixed_variances=30,
                environment_best_action_offset=300,
                action_offset=400,
                mul_factor=120,
                name="2c_dm_circ",
            ),
            CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                                                            + np.cos(time / 200)*100
                                                            + 0.5,
                fixed_variances=30,
                environment_best_action_offset=300,
                action_offset=400,
                mul_factor=120,
                name="2c_dm_fst_circ",
            ),
        ]

        default_actions_range = np.arange(0, 1_001, 100)
        action_bounds = (default_actions_range[0], default_actions_range[-1])
        default_steps_before_retraining_nn = 10
        reward_bounds = (0.0, 1.0)
        context_vector_size = 2
        pretrain_time = 10
        pretrain_policy = RandomPolicy(uniform(loc=0.5, scale=1_000))

        possible_policies = [
            UcbPolicy({k: v for k, v in enumerate(default_actions_range)}),
            UcbPolicy({k: v for k, v in enumerate(default_actions_range)}, sw=-200),
            ThompsonSamplingPolicy({k: v for k, v in enumerate(default_actions_range)}),
            ThompsonSamplingPolicy(
                {k: v for k, v in enumerate(default_actions_range)}, sw=-200
            ),
        ]

        # MEMF_NN_50_50_A01
        nn_layers = [50, 50]
        alpha = 0.1
        possible_policies.append(
            MaxEntropyModelFreeDiscrete(
                possible_actions=default_actions_range,
                name=f'MEMFD_NN_{[str(x) + "_" for x in nn_layers]}_a{str(alpha).replace(".", "")}',
                alpha_entropy=alpha,
                reward_estimator=LimitedNeuralNetworkRewardEstimator(
                    action_bounds=action_bounds,
                    reward_bounds=reward_bounds,
                    layers=nn_layers,
                    context_vector_size=context_vector_size,
                ),
                pretrain_time=pretrain_time,
                pretrain_policy=pretrain_policy,
            )
        )

        # MEMF_HMC_NN_50_50_A005
        nn_layers = [50, 50]
        alpha = 0.05
        possible_policies.append(
            MaxEntropyModelFreeContinuousHmc(
                mcmc_initial_state=500.0,
                name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}',
                alpha_entropy=alpha,
                reward_estimator=LimitedNeuralNetworkRewardEstimator(
                    action_bounds=action_bounds,
                    reward_bounds=reward_bounds,
                    layers=nn_layers,
                    context_vector_size=context_vector_size,
                ),
                pretrain_time=pretrain_time,
                pretrain_policy=pretrain_policy,
            )
        )

        # for mem in [200, 300]:
        for mem in [200]:
            # MEMF_NN_50_50_A01
            nn_layers = [50, 50]
            alpha = 0.1
            possible_policies.append(
                MaxEntropyModelFreeDiscrete(
                    possible_actions=default_actions_range,
                    name=f'MEMFD_NN_{[str(x) + "_" for x in nn_layers]}_a{str(alpha).replace(".", "")}_sig_{mem}',
                    alpha_entropy=alpha,
                    reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
                        short_term_mem=mem,
                        long_term_mem=mem,
                        action_bounds=action_bounds,
                        reward_bounds=reward_bounds,
                        layers=nn_layers,
                        context_vector_size=context_vector_size,
                    ),
                    pretrain_time=pretrain_time,
                    pretrain_policy=pretrain_policy,
                )
            )

            # MEMF_HMC_NN_50_50_A005
            nn_layers = [50, 50]
            alpha = 0.05
            possible_policies.append(
                MaxEntropyModelFreeContinuousHmc(
                    mcmc_initial_state=500.0,
                    name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}_sig_{mem}',
                    alpha_entropy=alpha,
                    reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
                        short_term_mem=mem,
                        long_term_mem=mem,
                        action_bounds=action_bounds,
                        reward_bounds=reward_bounds,
                        layers=nn_layers,
                        context_vector_size=context_vector_size,
                    ),
                    pretrain_time=pretrain_time,
                    pretrain_policy=pretrain_policy,
                )
            )


        special_policies = []

        nn_layers = [50, 50]
        alpha = 0.05
        mem = 200
        for num_burnin_steps in [50, 100, 500, 1_000]:
            for mcmc_initial_state,mis_id in zip([(0,1_000),"last_state"], ['rnd','ls_tst']):
                special_policies.append(
                    MaxEntropyModelFreeContinuousHmc(
                        mcmc_initial_state=mcmc_initial_state,
                        num_burnin_steps=num_burnin_steps,
                        name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}_sig_{mem}_nbs_{num_burnin_steps}_mis_{mis_id}',
                        alpha_entropy=alpha,
                        reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
                            short_term_mem=mem,
                            long_term_mem=mem,
                            action_bounds=action_bounds,
                            reward_bounds=reward_bounds,
                            layers=nn_layers,
                            context_vector_size=context_vector_size,
                        ),
                        pretrain_time=pretrain_time,
                        pretrain_policy=pretrain_policy,
                    )
                )

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
                    experiment_name="WINWIN" + environment.name,
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

        environment = possible_environments[-1]
        for policy_base in special_policies:
            policy = policy_base.__copy__()
            print(f"Running {policy.name} - {environment.name}")

            evaluator = Evaluator(
                run_name=f"{policy.name}",
                save_data=True,
                plot_data=False,
                use_mlflow=True,
                policy=policy,
                environment=environment,
                experiment_name="WINWIN" + environment.name,
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