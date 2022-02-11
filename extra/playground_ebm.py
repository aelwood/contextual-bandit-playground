from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random

from collections import defaultdict

import torch.optim as optim
from sklearn.datasets import make_blobs


class EnergyBasedModel(nn.Module):
    def __init__(self, in_features_size=32, hidden_feature_size=[32, 16, 8]):
        super(EnergyBasedModel, self).__init__()
        feat_sizes = [in_features_size] + hidden_feature_size

        cnn_layers = nn.ModuleList([])
        for i in range(len(feat_sizes) - 1):
            cnn_layers.append(torch.nn.Linear(feat_sizes[i], feat_sizes[i + 1]))
            cnn_layers.append(torch.nn.ReLU())
        cnn_layers.append(torch.nn.Linear(feat_sizes[-1], 1))

        self.layers = cnn_layers

    def reinitialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform(layer.weight)
                layer.bias.data.fill_(0.01)

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
        max_len=64,
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
            (
                torch.cat(
                    [torch.rand(context_size), torch.rand(1) * 10, torch.rand(1) > 0.5]
                ).unsqueeze(0)
            )
            for _ in range(self.sample_size)
        ]  # TODO: atm the fake examples are created with a context between 0 and 1
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
        if n_new == 0:
            n_new = 1
        rand_examples = torch.cat(
            [
                torch.cat(
                    [
                        torch.rand(self.context_size),
                        torch.rand(1) * 10,
                        torch.rand(1) > 0.5,
                    ]
                ).unsqueeze(0)
                for _ in range(n_new)
            ]
        )

        old_examples = torch.cat(
            random.choices(self.examples, k=self.sample_size - n_new), dim=0
        )
        inp_examples = (
            torch.cat([rand_examples, old_examples], dim=0).detach().to(self.device)
        )

        # Perform MCMC sampling
        inp_examples = LangevinDynamicsSampler.generate_samples(
            self.model, inp_examples, steps=steps, step_size=step_size
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples = (
            list(inp_examples.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.examples
        )
        self.examples = self.examples[: self.max_len]
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
            inp_features.grad.data.clamp_(
                -0.03, 0.03
            )  # For stabilizing and preventing too high gradients

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
            # inp_features.data.clamp_(min=0, max=10.0)

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
            # inp_features.data.clamp_(min=0, max=10.0)

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


if __name__ == "__main__":
    n_samples = 2_000
    n_features = 2
    centers = 2
    percentage_of_wrong_actions = 0.5

    metric_watcher = defaultdict(list)

    # # # # # # # # # # # # # # # #
    #        Dataset creation     #
    # # # # # # # # # # # # # # # #
    """
        This dataset is composed by two contex linearly separable (we are using the make_blobs function).
        Each context belongs to an action with the following boundaries: [0.7, 0.8], [0.2, 0.3]
    """
    context_vectors, context_ids = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=0.4,
        shuffle=True,
    )

    actions_a = np.random.uniform(
        low=0.7,
        high=0.8,
        size=(int(sum(context_ids == 0) * (1 - percentage_of_wrong_actions)),),
    )
    wrong_action_a = np.random.uniform(
        low=0.1,
        high=0.6,
        size=(int(sum(context_ids == 0) * percentage_of_wrong_actions),),
    )
    played_actions_a = np.hstack([actions_a, wrong_action_a])
    rewards_a = np.hstack([np.ones(len(actions_a)), np.zeros(len(wrong_action_a))])

    actions_b = np.random.uniform(
        low=0.2,
        high=0.3,
        size=(int(sum(context_ids == 1) * (1 - percentage_of_wrong_actions)),),
    )
    wrong_action_b = np.random.uniform(
        low=0.4,
        high=1.0,
        size=(int(sum(context_ids == 1) * percentage_of_wrong_actions),),
    )
    played_actions_b = np.hstack([actions_b, wrong_action_b])
    rewards_b = np.hstack([np.ones(len(actions_b)), np.zeros(len(wrong_action_b))])

    played_actions = np.zeros(len(context_ids))
    played_actions[context_ids == 0] = played_actions_a
    played_actions[context_ids == 1] = played_actions_b
    reward = np.zeros(len(context_ids))
    reward[context_ids == 0] = rewards_a
    reward[context_ids == 1] = rewards_b

    shuffling = np.arange(len(reward))
    np.random.shuffle(shuffling)

    context_vectors = context_vectors[shuffling]
    context_ids = context_ids[shuffling]
    played_actions = played_actions[shuffling]
    reward = reward[shuffling]

    plt.figure(figsize=(16, 9))

    x = np.arange(len(played_actions))
    plt.title("Actions")
    plt.plot(
        x[reward == 0],
        played_actions[reward == 0],
        label="played",
        linestyle="",
        marker="^",
        color="r",
    )
    plt.plot(
        x[reward == 1],
        played_actions[reward == 1],
        label="played",
        linestyle="",
        marker="*",
        color="b",
    )
    plt.show()

    past_contexts = context_vectors[: int(len(context_vectors) * 0.7)]
    past_actions = played_actions[: int(len(played_actions) * 0.7)]
    past_rewards = reward[: int(len(reward) * 0.7)]

    eval_past_contexts = context_vectors[
        : int(len(context_vectors) * 0.7) : int(len(context_vectors) * 0.8)
    ]
    eval_past_actions = played_actions[
        : int(len(played_actions) * 0.7) : int(len(played_actions) * 0.8)
    ]
    eval_past_rewards = reward[: int(len(reward) * 0.7) : int(len(reward) * 0.8)]

    test_past_contexts = context_vectors[int(len(context_vectors) * 0.8) :]
    test_past_actions = played_actions[int(len(played_actions) * 0.8) :]
    test_past_rewards = reward[int(len(reward) * 0.8) :]
    test_context_ids = context_ids[int(len(reward) * 0.8) :]

    """
    Training loop v1, no pretraining
    """
    num_epochs = 60
    sample_size = 256
    lr = 1e-5
    reg_lambda = 1
    steps = 100
    step_size = 0.1

    model = EnergyBasedModel(in_features_size=4)
    sampler = LangevinDynamicsSampler(
        model,
        (2,),
        sample_size,
        max_len=64,
        device=torch.device("cpu"),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 1, gamma=0.97
    )  # Exponential decay over epochs

    xp = np.hstack(
        [np.vstack(past_contexts), np.vstack(past_actions), np.vstack(past_rewards)]
    )
    xp = torch.tensor(xp)

    xp_eval = np.hstack(
        [
            np.vstack(eval_past_contexts),
            np.vstack(eval_past_actions),
            np.vstack(eval_past_rewards),
        ]
    )
    xp_eval = torch.tensor(xp_eval)

    for epoch in range(num_epochs):
        model.train()
        xp = xp[torch.randperm(xp.size()[0]), :]
        running_loss = []

        for xp_batch in xp.split(sample_size):
            xm_batch = sampler.sample_new_exmps(steps=steps, step_size=step_size)

            if not xp_batch.size(0) == xm_batch.size(0):
                xm_batch = xm_batch[torch.randperm(xp_batch.size()[0]), :]
                xm_batch = xm_batch[: xp_batch.size(0), :]

            optimizer.zero_grad()

            # forward + backward + optimize
            energy_xp = model(xp_batch)
            energy_xm = model(xm_batch)

            # Calculate losses
            reg_loss = reg_lambda * (energy_xp ** 2 + energy_xm ** 2).mean()
            cdiv_loss = energy_xm.mean() - energy_xp.mean()
            loss = reg_loss + cdiv_loss
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        scheduler.step()

        # Evaluation
        model.eval()
        eval_running_loss = []
        for xp_batch in xp_eval.split(sample_size):
            xm_batch = sampler.sample_new_exmps(steps=steps, step_size=step_size)

            if not xp_batch.size(0) == xm_batch.size(0):
                xm_batch = xm_batch[torch.randperm(xp_batch.size()[0]), :]
                xm_batch = xm_batch[: xp_batch.size(0), :]

            # forward + backward + optimize
            energy_xp = model(xp_batch)
            energy_xm = model(xm_batch)

            # Calculate losses
            reg_loss = reg_lambda * (energy_xp ** 2 + energy_xm ** 2).mean()
            cdiv_loss = energy_xm.mean() - energy_xp.mean()
            loss = reg_loss + cdiv_loss

            eval_running_loss.append(loss.item())

        scheduler.step()
        print(
            f"Epoch {epoch} avg training loss {np.mean(running_loss):0.4f} evaluation loss {np.mean(eval_running_loss):0.4f}"
        )
        metric_watcher["running_loss"].append(np.mean(running_loss))
        metric_watcher["eval_running_loss"].append(np.mean(eval_running_loss))

    for metric_name in metric_watcher.keys():
        plt.plot(
            np.arange(len(metric_watcher[metric_name])),
            metric_watcher[metric_name],
            label=metric_name,
        )

    plt.legend()
    plt.show()

    for _ in range(5):
        # TEST OF CREATING A NEW EXAMPLE
        context_a = test_past_contexts[test_context_ids == 1][0]

        action_to_play = torch.rand(1)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        action_to_play.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(action_to_play.shape, device=action_to_play.device) / 10

        # List for storing generations at each step (for later analysis)
        action_per_step = []
        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            action_to_play.data.add_(noise.data)

            # Part 2: calculate gradients for the current input.
            state = model(
                torch.cat(
                    [torch.tensor(context_a), action_to_play, torch.tensor([1])]
                ).float()
            )
            state.sum().backward()
            action_to_play.grad.data.clamp_(
                -0.03, 0.03
            )  # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            action_to_play.data.add_(-step_size * action_to_play.grad.data)
            action_to_play.grad.detach_()
            action_to_play.grad.zero_()
            # inp_features.data.clamp_(min=0, max=10.0)

            action_per_step.append(action_to_play.clone().detach())

        plt.plot(np.arange(len(action_per_step)), action_per_step, color="b")
        plt.plot(0, action_per_step[0], label="initial_state", marker="o")
        plt.plot(
            len(action_per_step) - 1, action_per_step[-1], label="final_state", marker="o"
        )
        plt.axhline(0.8, label="maximum", color="b")
        plt.axhline(0.7, label="minimun", color="r")
        plt.title("Optimal action investigation")
        plt.xlabel("Steps")
        plt.ylabel("Action")
        plt.legend()
        plt.show()
