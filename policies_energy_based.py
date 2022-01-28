import abc

from policies import PolicyABC


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

    def get_action(self, context):
        """
        here we want to draw a sample of the action to play, in Ashraf's Algo1 up to a <- ID(t)
        then we play this action (which happens in the simulate loop) and move on to train
        """
        # sample_probability_of_transition
        # calculate_expected_rewards_for_different_actions
        # find max reward, best_r, given action action
        self.expected_rewards_from_model.append(best_r)
        return action


    def get_params(self):
        pass
