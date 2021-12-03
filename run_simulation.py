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
)

from scipy.stats import uniform


def simulate(environment, policy, evaluator, evaluation_frequency=100):
    evaluation_frequency = evaluation_frequency
    for i, c in enumerate(environment.generate_contexts()):
        a = policy.get_action(c)
        r, s_r = environment.get_reward(a, c)
        policy.notify_event(c, a, s_r)

        optimal_r, optimal_a, stochastic_r = environment.get_best_reward_action(c)

        evaluator.notify(a, r, s_r, optimal_a, optimal_r, stochastic_r)
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


if __name__ == "__main__":
    run_ablation_test = True

    if not run_ablation_test:
        # single_context_static_reward_random_policy()
        # single_context_static_reward_cyclicexploration_policy()
        # single_context_static_reward_ucb_policy()
        # single_context_dynamic_reward_ucb_policy()
        # single_context_dynamic_reward_ucb_sw_policy() # ??? why
        # duouble_context_static_reward_ucb_policy()
        # duble_context_dynamic_reward_ucb_sw_policy()
        single_context_static_reward_LinUcbPolicy_policy()
        # double_context_static_reward_LinUcbPolicy_policy()
        # single_context_static_reward_hmc_policy()
        # single_context_static_reward_hmc_policy_nn()
        # single_context_static_reward_model_free_discrete_policy_nn()
        # double_context_static_reward_model_free_discrete_policy_nn()
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
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 1_000)
                + 0.5,
                name="1c_dm",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 1_000)
                + 0.5,
                name="2c_dm",
            ),
            SyntheticEnvironment(
                number_of_different_context=1,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 500)
                + 0.5,
                name="1c_dm_slw",
            ),
            SyntheticEnvironment(
                number_of_different_context=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu
                + np.cos(time / 500)
                + 0.5,
                name="2c_dm_slw",
            ),
        ]

        default_actions_range = np.arange(1, 8, 2)

        possible_policies = [
            UcbPolicy({k: v for k, v in enumerate(default_actions_range)}),
            UcbPolicy({k: v for k, v in enumerate(default_actions_range)}, sw=-200),
            ThompsonSamplingPolicy({k: v for k, v in enumerate(default_actions_range)}),
            # ThompsonSamplingPolicy(
            #     {k: v for k, v in enumerate(default_actions_range)}, sw=-200
            # ),
            # LinUcbPolicy({k: v for k, v in enumerate(default_actions_range)}, 3, 0.01),
        ]

        # for alpha in [0.5, 0.2, 0.1, 0.05, 0.02]:
        #     possible_policies.append(
        #         MaxEntropyModelFreeDiscrete(
        #             possible_actions=default_actions_range,
        #             name=f'MEMFD_a{str(alpha).replace(".","")}',
        #             alpha_entropy=alpha,
        #             reward_estimator=LimitedRidgeRegressionEstimator(
        #                 alpha_l2=1.0,
        #                 action_bounds=[0, 8],
        #                 reward_bounds=[0, 1],
        #             ),
        #             pretrain_time=10,
        #             pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
        #         )
        #     )
        #
        #     possible_policies.append(
        #         MaxEntropyModelFreeDiscrete(
        #             possible_actions=default_actions_range,
        #             name=f'MEMFD_MPA_a{str(alpha).replace(".","")}',
        #             alpha_entropy=alpha,
        #             reward_estimator=LimitedRidgeRegressionEstimatorModelPerArm(
        #                 actions=default_actions_range,
        #                 alpha_l2=1.0,
        #                 action_bounds=[0, 8],
        #                 reward_bounds=[0, 1],
        #             ),
        #             pretrain_time=50,
        #             pretrain_policy=CyclicExploration(default_actions_range),
        #         )
        #     )

        # possible_policies = []
        # for alpha in [0.5, 0.2, 0.1, 0.05, 0.02]:
        #     for nn_size in [5,10,20]:
        #         possible_policies.append(
        #         MaxEntropyModelFreeDiscrete(
        #             possible_actions=np.arange(0, 10, 1), # TODO: default_actions_range
        #             name=f'MEMFD_NN_{nn_size}_a{str(alpha).replace(".", "")}',
        #             alpha_entropy=alpha,
        #             reward_estimator=LimitedNeuralNetworkRewardEstimator(
        #                 action_bounds=(0.0, 10.0),
        #                 reward_bounds=(0.0, 1.0),
        #                 layers=[nn_size,nn_size],
        #                 context_vector_size=3,
        #             ),
        #             pretrain_time=10,
        #             pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
        #         ))


        # for alpha in [0.5, 0.2, 0.1, 0.05, 0.02]:
        #     policy = MaxEntropyModelFreeContinuousHmc(
        #         mcmc_initial_state=0.5,
        #         alpha_entropy=alpha,
        #         reward_estimator=LimitedNeuralNetworkRewardEstimator(
        #         action_bounds=(0.0, 10.0),
        #         reward_bounds=(0.0, 1.0),
        #         layers=[50, 50],
        #         context_vector_size=3,
        #     ),
        #         pretrain_time=10,
        #         pretrain_policy=RandomPolicy(uniform(loc=0.5, scale=10)),
        #     )







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
                    experiment_name=environment.name+'aaaa',
                )

                simulate(environment, policy, evaluator, evaluation_frequency=100)
