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
from policies_energy_based import EBMPolicy


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


def tes_EBM():
    # environment = CirclesSyntheticEnvironment(
    #             number_of_different_context=2,
    #             n_context_features=2,
    #             number_of_observations=2_000,
    #             time_perturbation_function=lambda time, mu: mu,
    #             fixed_variances=0.6,
    #             environment_best_action_offset=2,
    #             action_offset= 4,
    #             mul_factor= 1,
    #             name="2c_dm_fst_circ",
    #         )

    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=3_000,
        time_perturbation_function=lambda time, mu: mu,
        #fixed_variances=0.2,
        action_offset= 3
    )

    policy = EBMPolicy(
        name=f'EBM_new',
        warm_up=500,
        num_epochs=150,
        #loss_function_type="mce"
    )

    evaluator = Evaluator(
        run_name=f"{policy.name}_logloss_test",
        save_data=True,
        plot_data=True,
        use_mlflow=False,
        policy=policy,
        environment=environment,
        experiment_name=environment.name,
    )
    simulate(
        environment, policy, evaluator, evaluation_frequency=100, steps_to_train=256
    )


if __name__ == "__main__":
    run_ablation_test = False

    if not run_ablation_test:
        tes_EBM()
    else:
        # # NEW these are the ones with the big action offset
        # number_of_observations = 2_000
        # fixed_variances = 60 #0.6
        # environment_best_action_offset = 300 #2
        # action_offset = 400 #4
        # mul_factor = 100 #1
        # lambda_mul_factor = 100#1

        # OLD
        # number_of_observations = 2_000
        # fixed_variances = 0.6
        # environment_best_action_offset = 2
        # action_offset = 4
        # mul_factor = 1
        lambda_mul_factor =  1

        number_of_observations = 3000

        possible_environments = [
        #     SyntheticEnvironment(
        #         number_of_different_context=2,
        #         number_of_observations=3_000,
        #         time_perturbation_function=lambda time, mu: mu,
        #         # fixed_variances=0.2,
        #         action_offset=3,
        #         name="envlin"
        #     ),
        #     SyntheticEnvironment(
        #         number_of_different_context=2,
        #         number_of_observations=3_000,
        #         time_perturbation_function=lambda time, mu: mu
        #                                                      + np.cos(time / 500)*lambda_mul_factor
        #                                                      + 0.5,
        #         # fixed_variances=0.2,
        #         action_offset=3,
        #         name = "envlin_dyn"
        # ),
        #     SyntheticEnvironment(
        #         number_of_different_context=2,
        #         number_of_observations=3_000,
        #         time_perturbation_function=lambda time, mu: mu,
        #         # fixed_variances=0.2,
        #         action_offset=2,
        #         name = "envlin2"
        # ),
            CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu,
                # fixed_variances=fixed_variances,
                # environment_best_action_offset=environment_best_action_offset,
                action_offset=3,
                # mul_factor=mul_factor,
                circle_factor=4.,
                name="2c_4_circ",
            ),
            CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                                                        + np.cos(time / 500)*lambda_mul_factor
                                                        + 0.5,
                # fixed_variances=fixed_variances,
                # environment_best_action_offset=environment_best_action_offset,
                action_offset=3,
                # mul_factor=mul_factor,
                circle_factor=4.,
                name="2c_4_circ_dyn",
            ),
        ]

        # THESE ARE THE ENVIRONMENTS WE THOUGHT WE WOULD WIN ON....
        # super_envs = []
        #
        # # OLD
        # for variance,mul in zip([0.6,0.5,0.4,0.3,0.2,0.1],[1,1,0.9,0.7,0.4,0.2]):
        # # NEW
        # # for variance,mul in zip([60,50,40,30,20,10],[100,90,90,70,40,20]):
        #     super_envs.append(
        #     CirclesSyntheticEnvironment(
        #         number_of_different_context=2,
        #         n_context_features=2,
        #         number_of_observations=number_of_observations,
        #         time_perturbation_function=lambda time, mu: mu
        #                                                     + np.cos(time / 200) * lambda_mul_factor
        #                                                     + 0.5,
        #         fixed_variances=fixed_variances,
        #         environment_best_action_offset=environment_best_action_offset,
        #         action_offset=action_offset,
        #         mul_factor=mul_factor,
        #         name=f"2c_dm_fst_circ_{variance}_{mul}",
        #     ))



        # # THESE ARE THE WIDE ACTION RANGE STUFF THAT WE THOUG?HT WE WOULD WIN ON...
        # default_actions_range = np.arange(0, 1_001, 100) #np.arange(1, 10, 1)
        # action_bounds = (default_actions_range[0], default_actions_range[-1])
        # default_steps_before_retraining_nn = 10
        # reward_bounds = (0.0, 1.0)
        # context_vector_size = 2
        # pretrain_time = 10
        # pretrain_policy = RandomPolicy(uniform(loc=0.5, scale=1_000)) # RandomPolicy(uniform(loc=0.5, scale=10))
        # step_size=100 #1
        # mcmc_initial_state = 500.0 #5

        # OLD
        default_actions_range = np.arange(0.2, 6.2, 1)
        action_bounds = (default_actions_range[0], default_actions_range[-1])
        default_steps_before_retraining_nn = 100
        reward_bounds = (0.0, 1.0)
        context_vector_size = 3
        pretrain_time = 1000
        pretrain_policy = RandomPolicy(uniform(loc=0.5, scale=5))
        step_size=1
        mcmc_initial_state = 2.5

        baseline_policies = [
            UcbPolicy({k: v for k, v in enumerate(default_actions_range)}),
            UcbPolicy({k: v for k, v in enumerate(default_actions_range)}, sw=-200),
            ThompsonSamplingPolicy({k: v for k, v in enumerate(default_actions_range)}),
            ThompsonSamplingPolicy(
                {k: v for k, v in enumerate(default_actions_range)}, sw=-200
            ),
            LinUcbPolicy({k: v for k, v in enumerate(default_actions_range)}, context_vector_size, 0.05),
            LinUcbPolicy({k: v for k, v in enumerate(default_actions_range)}, context_vector_size, 0.1)
        ]

        algo_a_policies = []

        # # MEMF_NN_50_50_A01
        nn_layers = [50, 50]
        alpha = 0.1
        algo_a_policies.append(
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
        algo_a_policies.append(
            MaxEntropyModelFreeContinuousHmc(
                mcmc_initial_state=mcmc_initial_state,
                step_size=step_size,
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
            algo_a_policies.append(
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
            algo_a_policies.append(
                MaxEntropyModelFreeContinuousHmc(
                    mcmc_initial_state=mcmc_initial_state,
                    step_size=step_size,
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


        # special_policies = []
        #
        # nn_layers = [50, 50]
        # alpha = 0.05
        # mem = 200
        #
        # # OLD
        # for num_burnin_steps in [50, 100, 200]:
        #     for mcmc_initial_state,mis_id in zip([(0,10),"last_state"], ['rnd','ls_tst']):
        #         for step_size in [0.1,0.5, 1, 2, 5, 10]:
        #
        # # # NEW
        # # for num_burnin_steps in [50, 100, 300]:
        # #     for mcmc_initial_state,mis_id in zip([(0,1_000),"last_state"], ['rnd','ls_tst']):
        # #         for step_size in [50, 75, 100, 125, 150]:
        #             special_policies.append(
        #                 MaxEntropyModelFreeContinuousHmc(
        #                     mcmc_initial_state=mcmc_initial_state,
        #                     num_burnin_steps=num_burnin_steps,
        #                     step_size=step_size,
        #                     name=f'MEMF_HMC_NN_{"_".join([str(x) for x in nn_layers])}_a{str(alpha).replace(".", "")}_sig_{mem}_nbs_{num_burnin_steps}_mis_{mis_id}_step_{step_size}',
        #                     alpha_entropy=alpha,
        #                     reward_estimator=LimitedNeuralNetworkRewardEstimatorTrainingWeightsSigmoid(
        #                         short_term_mem=mem,
        #                         long_term_mem=mem,
        #                         action_bounds=action_bounds,
        #                         reward_bounds=reward_bounds,
        #                         layers=nn_layers,
        #                         context_vector_size=context_vector_size,
        #                     ),
        #                     pretrain_time=pretrain_time,
        #                     pretrain_policy=pretrain_policy,
        #                 )
        #             )

        algo_b_policies = [
            # EBMPolicy(
            #     name=f'EBM_NN_baseline',
            #     warm_up=pretrain_time,
            #     num_epochs=150,
            #     loss_function_type="log",
            # ),
            # EBMPolicy(
            #     name=f'EBM_NN_baseline',
            #     warm_up=pretrain_time,
            #     num_epochs=150,
            #     loss_function_type="mce", # FIXME this doesn't work
            # )
            EBMPolicy(
                name=f'EBM_NN_circ_hp_q',
                lr=0.005,
                warm_up=pretrain_time,
                num_epochs=150,
                loss_function_type="log",
                sample_size=256,
                output_quadratic=True,
            ),
            EBMPolicy(
                name=f'EBM_NN_circ_hp_l',
                lr=0.005,
                warm_up=pretrain_time,
                num_epochs=150,
                loss_function_type="log",
                sample_size=256,
                output_quadratic=False,
            ),
        ]

        policies_to_run = baseline_policies + algo_b_policies + algo_a_policies
        #policies_to_run = algo_b_policies
        for policy_base in policies_to_run:
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
                    experiment_name="inc_ebm_" + environment.name,
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

        # BELOW HERE WERE THE SPECIAL CASES WE WERE MAKING TO TRY TO WIN:
        # environment = possible_environments[-1]
        # for policy_base in special_policies:
        #     policy = policy_base.__copy__()
        #     print(f"Running {policy.name} - {environment.name}")
        #
        #     evaluator = Evaluator(
        #         run_name=f"{policy.name}",
        #         save_data=True,
        #         plot_data=False,
        #         use_mlflow=True,
        #         policy=policy,
        #         environment=environment,
        #         experiment_name="HOPE" + environment.name,
        #     )
        #
        #     steps_to_train = 1
        #     if "NN" in policy.name:
        #         steps_to_train = default_steps_before_retraining_nn
        #
        #
        #     simulate(
        #         environment,
        #         policy,
        #         evaluator,
        #         evaluation_frequency=100,
        #         steps_to_train=steps_to_train,
        #     )
        #
        #
        # for env in super_envs:
        #     policy = possible_policies[-1].__copy__()
        #     print(f"Running {policy.name} - {env.name}")
        #
        #     evaluator = Evaluator(
        #         run_name=f"{policy.name}",
        #         save_data=True,
        #         plot_data=False,
        #         use_mlflow=True,
        #         policy=policy,
        #         environment=env,
        #         experiment_name="VARING_env_std_and_mul",
        #     )
        #
        #     steps_to_train = 1
        #     if "NN" in policy.name:
        #         steps_to_train = default_steps_before_retraining_nn
        #
        #     simulate(
        #         environment,
        #         policy,
        #         evaluator,
        #         evaluation_frequency=100,
        #         steps_to_train=steps_to_train,
        #     )