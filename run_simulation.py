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

# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

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


if __name__ == "__main__":
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

    lambda_mul_factor = 1
    number_of_observations = 10_000

    possible_environments = [
    #     SyntheticEnvironment(
    #         number_of_different_context=2,
    #         number_of_observations=number_of_observations,
    #         time_perturbation_function=lambda time, mu: mu,
    #         # fixed_variances=0.2,
    #         action_offset=3,
    #         name="envlin"
    #     ),
    #     SyntheticEnvironment(
    #         number_of_different_context=2,
    #         number_of_observations=number_of_observations,
    #         time_perturbation_function=lambda time, mu: mu
    #                                                      + np.cos(time / 500)*lambda_mul_factor
    #                                                      + 0.5,
    #         # fixed_variances=0.2,
    #         action_offset=3,
    #         name = "envlin_dyn"
    # ),
    #     SyntheticEnvironment(
    #         number_of_different_context=2,
    #         number_of_observations=number_of_observations,
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


    # OLD
    default_actions_range = np.arange(0.2, 6.2, 1)
    action_bounds = (default_actions_range[0], default_actions_range[-1])
    default_steps_before_retraining_nn = 100
    reward_bounds = (0.0, 1.0)
    context_vector_size = 2 # TODO pay attention to this, it's different for circles and lin sep
    pretrain_time = 1000
    pretrain_policy = RandomPolicy(uniform(loc=0.5, scale=5))
    step_size=1
    mcmc_initial_state = 2.5

    baseline_policies = [
        UcbPolicy({k: v for k, v in enumerate(default_actions_range)}),
        ThompsonSamplingPolicy({k: v for k, v in enumerate(default_actions_range)}),
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


    algo_b_policies = []
    # alpha = 10
    # algo_b_policies.append(EBMPolicy(
    #     name=f'EBM_NN_circ_hp_l_a_{alpha}',
    #     lr=0.005,
    #     warm_up=pretrain_time,
    #     num_epochs=150,
    #     loss_function_type="log",
    #     sample_size=256,
    #     output_quadratic=False,
    #     alpha=alpha,
    #     feature_size=context_vector_size,
    #  ))

    for alpha in [ 1, 10, 100]:
         algo_b_policies.append(EBMPolicy(
             name=f'EBM_NN_circ_hp_q_a_{alpha}',
             lr=0.005,
             warm_up=pretrain_time,
             num_epochs=150,
             loss_function_type="log",
             sample_size=256,
             output_quadratic=True,
             alpha=alpha,
             feature_size = context_vector_size
         ))



    policies_to_run = algo_b_policies# + algo_a_policies + baseline_policies

    for policy_base in policies_to_run:
        for x in range(10):
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
                    experiment_name="CANO_" + environment.name,
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
