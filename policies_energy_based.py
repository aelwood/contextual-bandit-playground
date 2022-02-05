from policies import PolicyABC

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EnergyBasedModel(nn.Module):
    def __init__(self, in_features_size=32, hidden_feature_size=[128, 64, 32]):
        super(EnergyBasedModel, self).__init__()
        feat_sizes = [in_features_size] + hidden_feature_size

        cnn_layers = nn.ModuleList([])
        for i in range(len(feat_sizes) - 1):
            cnn_layers.append(torch.nn.Linear(feat_sizes[i], feat_sizes[i + 1]))
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
    def __init__(
        self,
        model,
        context_size: tuple,
        sample_size,
        max_len=8192,
        device=torch.device("cpu"),
    ):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.context_size = context_size
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [
            (torch.cat([torch.rand(context_size),torch.rand(1)*10, torch.rand(1)>0.5]).unsqueeze(0)) for _ in range(self.sample_size)
        ] # TODO: atm the fake examples are created with a context between 0 and 1
        self.device = device


    def sample_new_exmps(self, steps=60, step_size=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        if n_new==0:
            n_new=1
        rand_examples = torch.cat([
            torch.cat([torch.rand(self.context_size), torch.rand(1) * 10, torch.rand(1) > 0.5]).unsqueeze(0) for _ in
            range(n_new)
        ])

        old_examples = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_examples = torch.cat([rand_examples, old_examples], dim=0).detach().to(self.device)

        # Perform MCMC sampling
        inp_examples = LangevinDynamicsSampler.generate_samples(self.model, inp_examples, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_examples.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_examples


    @staticmethod
    def generate_samples(
        model, inp_features, steps=60, step_size=10, return_img_per_step=False
    ):
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
            out_imgs = -model(inp_features)
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

    @staticmethod
    def generate_samples_from_context(
        model, context, steps=60, step_size=10, return_img_per_step=False
    ):
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
        inp_features = torch.rand(1) * 10

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
            out_imgs = model(
                torch.cat(
                    [torch.tensor(context), inp_features, torch.tensor([1])]
                ).float()
            )
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
    def __init__(
        self,
        name="EnergyBasedModel",
        feature_size=2,
        ebm_estimator_class=EnergyBasedModel,
        sampler_class=LangevinDynamicsSampler,
        reg_lambda=1,
        lr=0.0005,
        num_epochs=50,
        sample_size=32,
        warm_up=64,
        device=torch.device("cpu"),
    ):
        self.past_rewards = []
        self.past_actions = []
        self.expected_rewards_from_model = []
        self.past_contexts = []
        assert warm_up > sample_size
        self.warm_up = warm_up
        self.sample_size = sample_size

        self.adjusted_feat_size = feature_size + 2  # action + reward


        self.name = name
        self.num_epochs = num_epochs
        self.ebm_estimator = ebm_estimator_class(
            in_features_size=self.adjusted_feat_size
        )

        self.sampler = sampler_class(
            self.ebm_estimator,
            (feature_size,),
            sample_size,
            max_len=8192,
            device=device,
        )
        self.reg_lambda = reg_lambda
        self.optimizer = optim.Adam(self.ebm_estimator.parameters(), lr=lr)

    def notify_event(self, context, action, stochastic_reward):
        self.past_contexts.append(context)
        self.past_rewards.append(int(stochastic_reward))
        self.past_actions.append(action)

    def train(self):
        """
        here we want to implement the second half of Algo1
        beta pos = self.past_contexts, self.past_rewards
        beta neg = self.past_contexts, self.expected_rewards_from_model
        """
        if self.warm_up > len(self.past_contexts):
            return None
        xp = np.hstack([np.vstack(self.past_contexts), np.vstack(self.past_actions), np.vstack(self.past_rewards)])
        xp = torch.tensor(xp)

        # do the gradient update step

        for epoch in range(self.num_epochs):
            self.ebm_estimator.train()
            xp = xp[torch.randperm(xp.size()[0]), :]  # random shuffling
            losses = []
            for xp_batch in xp.split(self.sample_size):
                xm_batch = self.sampler.sample_new_exmps(steps=60, step_size=10)

                if not xp_batch.size(0) == xm_batch.size(0):
                    xm_batch = xm_batch[torch.randperm(xp_batch.size()[0]), :]
                    xm_batch = xm_batch[:xp_batch.size(0),:]
                #TODO: uniform dimension

                self.optimizer.zero_grad()
                # forward + backward + optimize
                energy_xp = self.ebm_estimator(xp_batch)
                energy_xm = self.ebm_estimator(xm_batch)

                # Calculate losses
                reg_loss = self.reg_lambda * (energy_xp ** 2 + energy_xm ** 2).mean()
                cdiv_loss = energy_xm.mean() - energy_xp.mean()
                loss = reg_loss + cdiv_loss
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            print(f'Epoch {epoch} avg loss {np.mean(losses)}')

        self.ebm_estimator.eval()
        #TODO: maybe valdiation

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

        # TODO: sample with SGLD or MPPI instead?
        action = self.sampler.generate_samples_from_context(self.ebm_estimator, context)
        expected_reward = self.ebm_estimator(
            torch.cat([torch.tensor(context), action, torch.tensor([1])]).float()
        )
        self.expected_rewards_from_model.append(expected_reward.item())
        return float(action)

    def get_params(self):
        pass

    # OLD TRAINING PROCEDURE
    # def train(self):
    #     """
    #     here we want to implement the second half of Algo1
    #     beta pos = self.past_contexts, self.past_rewards
    #     beta neg = self.past_contexts, self.expected_rewards_from_model
    #     """
    #
    #     # do the gradient update step
    #     self.ebm_estimator.train()
    #
    #     xp = (self.past_contexts[-1], self.past_actions[-1], int(self.past_rewards[-1]))
    #     xm = (self.past_contexts[-1], self.past_actions[-1], self.expected_rewards_from_model[-1])
    #     xp = torch.tensor(np.hstack(xp))
    #     xm = torch.tensor(np.hstack(xm))
    #
    #
    #
    #     self.optimizer.zero_grad()
    #
    #     # forward + backward + optimize
    #     energy_xp = self.ebm_estimator(xp)
    #     energy_xm = self.ebm_estimator(xm)
    #
    #     # Calculate losses
    #     reg_loss = self.reg_lambda * (energy_xp ** 2 + energy_xm ** 2).mean()
    #     cdiv_loss = energy_xm.mean() - energy_xp.mean()
    #     loss = reg_loss + cdiv_loss
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     self.ebm_estimator.eval()
