from policies import PolicyABC

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#  TODO: LATER DISCUSS the reward should not be here
class EnergyBasedModel(nn.Module):
    def __init__(self, in_features_size=32, hidden_feature_size=[128, 64, 32]):
        super(EnergyBasedModel, self).__init__()
        feat_sizes = [in_features_size] + hidden_feature_size

        cnn_layers = nn.ModuleList([])
        for i in range(len(feat_sizes)-1):
            cnn_layers.append(torch.nn.Linear(feat_sizes[i], feat_sizes[i+1]))
            cnn_layers.append(torch.nn.ReLU())
        cnn_layers.append(torch.nn.Linear(feat_sizes[-1], 1))

        self.layers = cnn_layers

    def forward(self, x):
        # x is [context, action, reward]
        # y = torch.cat(x)
        y = x.float()
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        energy = y.squeeze(dim=-1)
        return energy


class LangevinDynamicsSampler:
    def __init__(self, model, feature_size:tuple, sample_size, max_len=8192, device=torch.device("cpu") ):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.feature_size = feature_size
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand(feature_size)*2-1) for _ in range(self.sample_size)]
        self.device = device
        # TODO: last element of each sample should be integer

    @staticmethod
    def generate_samples(model, context, steps=60, step_size=10, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.

        #  Context, action, state
        inp_features = torch.rand(1)*10

        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_features.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_features.shape, device=inp_features.device)

        # List for storing generations at each step (for later analysis)
        feature_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_features.data.add_(noise.data)
            inp_features.data.clamp_(min=0, max=10.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = model(torch.cat([torch.tensor(context), inp_features, torch.tensor([1])]).float())
            out_imgs.sum().backward()
            # inp_features.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_features.data.add_(-step_size * inp_features.grad.data)
            inp_features.grad.detach_()
            inp_features.grad.zero_()
            inp_features.data.clamp_(min=0, max=10.0)

            if return_img_per_step:
                feature_per_step.append(inp_features.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(feature_per_step, dim=0)
        else:
            return inp_features


class EBMPolicy(PolicyABC):
    def __init__(self, name="EnergyBasedModel",feature_size = 2, ebm_estimator_class=EnergyBasedModel, sampler_class = LangevinDynamicsSampler, reg_lambda=1, lr=0.05, device=torch.device("cpu")):
        self.past_rewards = []
        self.past_actions = []
        self.expected_rewards_from_model = []
        self.past_contexts = []

        self.adjusted_feat_size = feature_size + 2  # action + reward
        sample_size = 32

        self.name = name
        self.ebm_estimator = ebm_estimator_class(in_features_size=self.adjusted_feat_size)
        self.sampler_class = sampler_class(self.ebm_estimator,(self.adjusted_feat_size,), sample_size, max_len=8192, device=device)
        self.reg_lambda = reg_lambda
        self.optimizer = optim.Adam(self.ebm_estimator.parameters(), lr=lr)

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
        # TODO: do we wanna build the batch from previous samples
        self.ebm_estimator.train()
        xp = (self.past_contexts[-1], self.past_actions[-1], int(self.past_rewards[-1]))
        xm = (self.past_contexts[-1], self.past_actions[-1], self.expected_rewards_from_model[-1])
        xp = torch.tensor(np.hstack(xp))
        xm = torch.tensor(np.hstack(xm))

        self.optimizer.zero_grad()
        # (torch.cat([torch.tensor(context), action, torch.tensor([1])]).float())

        # forward + backward + optimize
        energy_xp = self.ebm_estimator(xp)
        energy_xm = self.ebm_estimator(xm)

        # Calculate losses
        reg_loss = self.reg_lambda * (energy_xp ** 2 + energy_xm ** 2).mean()
        cdiv_loss = energy_xm.mean() - energy_xp.mean()
        loss = reg_loss + cdiv_loss
        loss.backward()
        self.optimizer.step()

        self.ebm_estimator.eval()

    # def _get_mcmc_kernel(self, log_prob_function):
    #     return tfp.mcmc.HamiltonianMonteCarlo(
    #         target_log_prob_fn=log_prob_function, step_size=self.step_size, num_leapfrog_steps=2
    #     )

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

        # e = self.ebm_estimator.predict_prob_maintaining_graph

        # def unnormalized_log_prob(a):
        #     return e(context, a, r=1)

        # action = tfp.mcmc.sample_chain(
        #     num_results=1,
        #     num_burnin_steps=self.num_burnin_steps,
        #     current_state=self._get_mcmc_initial_state(),
        #     kernel=self._get_mcmc_kernel(log_prob_function=unnormalized_log_prob),
        #     trace_fn=None,
        # )

        # TODO: sample with SGLD or MPPI instead?
        action = self.sampler_class.generate_samples(self.ebm_estimator, context)
        expected_reward = self.ebm_estimator(torch.cat([torch.tensor(context), action, torch.tensor([1])]).float())
        self.expected_rewards_from_model.append(expected_reward.item())
        return float(action)

    def get_params(self):
        pass




if __name__ == "__main__":
    ciao = EnergyBasedModel()
    ciapne = ciao([torch.rand(30), torch.rand(1), torch.tensor([1])])



# TODO: model train
# TODO: model eval