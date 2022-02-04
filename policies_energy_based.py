import abc

from policies import PolicyABC
import tensorflow_probability as tfp


class ModelBasedABC(PolicyABC, metaclass=abc.ABCMeta):
    def __init__(self):
        self.past_rewards = []
        self.past_actions = []
        self.expected_rewards_from_model = []
        self.past_contexts = []

    def notify_event(self, context, action, stochastic_reward):
        self.past_contexts.append(context)
        self.past_rewards.append(stochastic_reward)
        self.past_actions.append(action)

    def train(self):
        """
        here we want to implement the second half of Algo1
        beta pos = self.past_contexts, self.past_rewards
        beta neg = self.past_contexts, self.expected_rewards_from_model
        """

        # do the gradient update step
        # TODO; understand if we want more sophisticated sampling here?
        xp = (self.past_contexts[-1], self.past_actions[-1], self.past_rewards[-1])
        xm = (self.past_contexts[-1], self.past_actions[-1], self.expected_rewards_from_model[-1])

        # now do the gradient update step

    def _get_mcmc_kernel(self, log_prob_function):
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_function, step_size=self.step_size, num_leapfrog_steps=2
        )

    def get_action(self, context):
        """
        here we want to draw a sample of the action to play, in Ashraf's Algo1 up to a <- ID(t)
        then we play this action (which happens in the simulate loop) and move on to train
        """

        # E(s_t, s_t+1) -> E(c, a, r)
        # the energy defines the probability of a particular triplet (c,a,r)
        # you have c (the input) and r (the goal state)
        # you want to draw a sample from the action to give the goal state (r=1) given c

        # i.e. the probability of getting a reward given this context and action
        # r = 0 or 1
        # our goal state is r=1 (G in Ashraf's paper)
        # generate sample p(t|s_i, G) -> p(a | c, r=1)
        # this gives us our action

        # NOTE: e, is equivalent to r in our Model Free implementation, but...
        # they are trained in a different way

        e = self.ebm_estimator.predict_prob_maintaining_graph

        def unnormalized_log_prob(a):
            return e(context, a, r=1)

        # TODO: sample with SGLD or MPPI instead?
        action = tfp.mcmc.sample_chain(
            num_results=1,
            num_burnin_steps=self.num_burnin_steps,
            current_state=self._get_mcmc_initial_state(),
            kernel=self._get_mcmc_kernel(log_prob_function=unnormalized_log_prob),
            trace_fn=None,
        )
        self.expected_rewards_from_model.append(action)
        return float(action)

    def get_params(self):
        pass
