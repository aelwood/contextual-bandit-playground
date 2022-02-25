from policies import PolicyABC

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class EnergyBasedModel(nn.Module):
    def __init__(self, in_features_size=32, hidden_feature_size=[32, 16, 8]):
        super(EnergyBasedModel, self).__init__()
        feat_sizes = [in_features_size] + hidden_feature_size

        cnn_layers = nn.ModuleList([
            torch.nn.Linear(in_features_size, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 1),
        ])

        self.layers = cnn_layers

    def reinitialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x, y):
        """
        :param x: [context,reward]
        :param y: [action]
        :return: energy
        """
        y = y
        x = x.float()
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.squeeze(dim=-1)
        energy = 1 / 2 * torch.pow(x - y, 2)
        return energy


class EBMPolicy(PolicyABC):
    def __init__(
        self,
        name="EnergyBasedModel",
        feature_size=2,
        ebm_estimator_class=EnergyBasedModel,
        lr=1e-3,
        warm_up=64,
        device=torch.device("cpu"),
    ):
        self.past_rewards = []
        self.past_actions = []
        self.past_contexts = []
        self.warm_up = warm_up
        self.sample_size = 32
        self.last_training_idx = 0

        self.adjusted_feat_size = feature_size + 1  # + reward

        self.name = name

        self.ebm_estimator = ebm_estimator_class(
            in_features_size=self.adjusted_feat_size
        )
        self.optimizer = optim.Adam(self.ebm_estimator.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

    def notify_event(self, context, action, stochastic_reward):
        self.past_contexts.append(context)
        self.past_rewards.append(int(stochastic_reward))
        self.past_actions.append(action)

    def train(self):
        sample_size = 32
        if self.warm_up > len(self.past_contexts):
            return None
        self.ebm_estimator.reinitialize_weights()
        for epoch in range(32):
            context_to_train = self.past_contexts[self.last_training_idx:]
            actions_to_train = self.past_actions[self.last_training_idx:]
            rewards_to_train = self.past_rewards[self.last_training_idx:]

            self.ebm_estimator.train()

            xp = np.hstack(
                [np.vstack(context_to_train), np.vstack(rewards_to_train)]
            )
            xp = torch.tensor(xp)
            yp = torch.FloatTensor(actions_to_train)

            running_loss = []
            for xp_batch, yp_batch in zip(xp.split(sample_size), yp.split(sample_size)):
                xp_batch.requires_grad=True
                yp_batch.requires_grad=True
                self.optimizer.zero_grad()
                y = self.ebm_estimator(xp_batch, yp_batch)
                loss = y.mean()
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())

            print(
                f"Epoch {epoch} avg training loss {np.mean(running_loss):0.4f}"
            )
            self.scheduler.step()

    def get_action(self, context):
        if self.warm_up >= len(self.past_contexts):
            return np.random.rand()*10
        steps = 100
        step_size = 0.2
        taken_context_plus_reward = torch.FloatTensor(np.hstack((context, [1])))

        action_to_play = torch.rand(1)

        self.ebm_estimator.eval()
        for p in self.ebm_estimator.parameters():
            p.requires_grad = False
        action_to_play.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(action_to_play.shape, device=action_to_play.device)

        # List for storing generations at each step (for later analysis)
        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            action_to_play.data.add_(noise.data)

            # Part 2: calculate gradients for the current input.
            state = -self.ebm_estimator(taken_context_plus_reward, action_to_play)
            state.sum().backward()

            # Apply gradients to our current samples
            action_to_play.data.add_(step_size * action_to_play.grad.data)
            action_to_play.grad.detach_()
            action_to_play.grad.zero_()
            # inp_features.data.clamp_(min=0, max=10.0)

        return float(action_to_play.clone().detach().item())

    def get_params(self):
        pass