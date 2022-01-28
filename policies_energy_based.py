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


    def _significantly_deviated(self, past_rewards, expected_rewards_from_model )->bool:
        """
            compare self.past_rewards vs self.expected_rewards_from_model
        """



    def train(self):
        """
        here we want to implement the second half of Algo1
        beta pos = self.past_contexts, self.past_rewards
        beta neg = self.past_contexts, self.expected_rewards_from_model
        """
        if self._significantly_deviated(self.past_rewards, self.expected_rewards_from_model):
            self.nn.train(
                self.past_contexts, self.past_rewards, self.past_actions, self.expected_rewards_from_model
            )
        else:
            pass


    def _get_mcmc_kernel(self, log_prob_function):
        return tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_function, step_size=self.step_size, num_leapfrog_steps=2
        )

    def linear_reacher_inverse_dynamics(self,trajectory):
        """
         In order to get the action from the state we have to implement
        https://github.com/yilundu/model_based_planning_ebm/blob/master/train.py#L316
         (linear_reacher_inverse_dynamics)
        """

    def get_action(self, context):
        """
        here we want to draw a sample of the action to play, in Ashraf's Algo1 up to a <- ID(t)
        then we play this action (which happens in the simulate loop) and move on to train
        """
        # sample_probability_of_transition
        # calculate_expected_rewards_for_different_actions
        # find max reward, best_r, given action action
        # self.expected_rewards_from_model.append(best_r)
        # return action
        def get_nn_distribution(curtent_state, goal_state, context):
            return self.nn(curtent_state, goal_state, context)

        trajectory = tfp.mcmc.sample_chain(
            num_results=1,
            num_burnin_steps=self.num_burnin_steps,
            current_state=self._get_mcmc_initial_state(),
            kernel=self._get_mcmc_kernel(log_prob_function=get_nn_distribution),
            trace_fn=None,
        )
        self.expected_rewards_from_model.append(trajectory)
        state = self.linear_reacher_inverse_dynamics(trajectory)
        return float(state)


    def get_params(self):
        pass
