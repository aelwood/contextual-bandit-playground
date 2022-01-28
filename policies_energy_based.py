import abc

from policies import PolicyABC

class ModelBasedABC(PolicyABC, metaclass=abc.ABCMeta):
    def __init__(self):
        self.past_rewards = []
        self.past_actions = []
        self.past_contexts = []

    def notify_event(self, context, action, stochastic_reward):
        self.past_contexts.append(context)
        self.past_rewards.append(stochastic_reward)
        self.past_actions.append(action)


    def train(self):
       pass

    def get_action(self, context):
        pass

    def get_params(self):
        pass
