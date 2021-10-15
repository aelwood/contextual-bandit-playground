import numpy as np

from environments import SyntheticEnvironment
from evaluator import Evaluator
from policies import RandomPolicy, UcbPolicy

from scipy.stats import uniform


def simulate(environment, policy, evaluator, evaluation_frequency=100):
    evaluation_frequency = evaluation_frequency
    for i, c in enumerate(environment.generate_contexts()):
        a = policy.get_action(c)
        r, s_r = environment.get_reward(a, c)
        policy.notify_event(c, a, s_r)

        optimal_r, optimal_a, stochastic_r = environment.get_best_reward_action(c)

        evaluator.notify(a, r, s_r, optimal_a, optimal_r,stochastic_r)
        policy.train()
        if i % evaluation_frequency == 0 and i > 0:
            print(evaluator.get_stats())


    print("Final results")
    print(evaluator.get_stats())
    evaluator.plot()


def single_context_static_reward_random_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
    )

    policy = RandomPolicy(uniform(loc=0.5, scale=10))
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)

def single_context_static_reward_ucb_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(1, 6, 2))})
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)

def single_context_dynamic_reward_ucb_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0., 3, 0.5))})
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)

def single_context_dynamic_reward_ucb_sw_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=1,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0., 3, 0.5))},sw=-200)
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)

def duble_context_dynamic_reward_ucb_sw_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu + np.cos(time / 500) + 0.5,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0., 3, 0.5))},sw=-200)
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)

def duouble_context_static_reward_ucb_policy():
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=2_000,
        time_perturbation_function=lambda time, mu: mu,
    )
    policy = UcbPolicy({k: v for k, v in enumerate(np.arange(0.5, 3, 0.5))})
    evaluator = Evaluator()

    simulate(environment, policy, evaluator, evaluation_frequency=100)

if __name__ == "__main__":
    # single_context_static_reward_random_policy()
    # single_context_static_reward_ucb_policy()
    # single_context_dynamic_reward_ucb_policy()
    single_context_dynamic_reward_ucb_sw_policy()
    # duble_context_dynamic_reward_ucb_sw_policy()
    # duouble_context_static_reward_ucb_policy()


