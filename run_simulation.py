from environments import SyntheticEnvironment

if __name__ == "__main__":
    environment = SyntheticEnvironment(
        number_of_different_context=2,
        number_of_observations=200,
        # time_perturbation_function=lambda time, mu: mu + (time // 100) * 5,
        time_perturbation_function=lambda time, mu: mu + np.cos(time/500),
    )

    policy = UcbPolicy()
    warmup_policy = RandomPolicy()
    # evaluator = Evaluator()  # Can have MLflow integrated into it
    # Linear contextual bandit

    warmup_length = 100

    evaluation_frequency = 100

    for i, c in enumerate(environment.generate_contexts()):

        if i < warmup_length:
            a = warmup_policy.get_action(c)
            r = environment.get_reward(a, c)
            r = environment.get_reward(10, c)
            policy.notify_event(c, r)
            warmup_policy.notify_event(c, r)

        else:

            policy.train()
            a = policy.get_action(c)
            r = environment.get_reward(a, c)
            policy.notify_event(c, r)

            optimal_r, optimal_a = environment.get_best_reward_action(c)

            evaluator.notify(a, r, optimal_a, optimal_r)

            if i % evaluation_frequency == 0 and i > warmup_length:
                print(evaluator.get_stats())

    print("Final results")
    print(evaluator.get_stats())
