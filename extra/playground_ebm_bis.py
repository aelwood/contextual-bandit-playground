from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random

from collections import defaultdict

import torch.optim as optim
from sklearn.datasets import make_blobs, make_circles
import matplotlib.animation as animation

## setting the random seeds, for easy testing and developments
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

"""
Investigation shit
"""
class EBMModel(nn.Module):
    def __init__(self, in_features_size=32, hidden_feature_size=[32, 16, 8]):
        super(EBMModel, self).__init__()
        feat_sizes = [in_features_size] + hidden_feature_size

        # cnn_layers = nn.ModuleList([])
        # for i in range(len(feat_sizes) - 1):
        #     cnn_layers.append(torch.nn.Linear(feat_sizes[i], feat_sizes[i + 1]))
        #     cnn_layers.append(torch.nn.ReLU())
        # cnn_layers.append(torch.nn.Linear(feat_sizes[-1], 1))

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
        x = x.float()
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.squeeze(dim=-1)
        energy = 1/2 * torch.pow(x-y,2)
        return energy


if __name__ == "__main__":
    n_samples = 2_000
    n_features = 2
    centers = 2
    percentage_of_wrong_actions = 0.5

    # actions_a_range = (0.7, 0.8)
    # actions_a_wrong_range = (0.1, 0.6)
    # actions_b_range = (0.2, 0.3)
    # actions_b_wrong_range = (0.4, 0.9)
    # plotting_range = (0, 1)

    # Is the action rage somehow important? NO
    actions_a_range = (10, 12)
    actions_a_wrong_range = (1, 8)
    actions_b_range = (2, 4)
    actions_b_wrong_range = (6, 13)

    plotting_range = (1,20)
    positive_reward = 1
    negative_reward = -1

    # # # # # # # # # # # # # # # #
    #        Dataset creation     #
    # # # # # # # # # # # # # # # #
    """
        This dataset is composed by two contex linearly separable (we are using the make_blobs function).
        Each context belongs to an action with the following boundaries: [0.7, 0.8], [0.2, 0.3]
    """

    # context_vectors, context_ids = make_blobs(
    #     n_samples=n_samples,
    #     n_features=n_features,
    #     centers=centers,
    #     cluster_std=0.4,
    #     shuffle=True,
    # )

    context_vectors, context_ids = make_circles(
        n_samples=n_samples,
        shuffle=True,
    )

    actions_a = np.random.uniform(
        low=actions_a_range[0],
        high=actions_a_range[1],
        size=(int(sum(context_ids == 0) * (1 - percentage_of_wrong_actions)),),
    )
    wrong_action_a = np.random.uniform(
        low=actions_a_wrong_range[0],
        high=actions_a_wrong_range[1],
        size=(int(sum(context_ids == 0) * percentage_of_wrong_actions),),
    )
    played_actions_a = np.hstack([actions_a, wrong_action_a])
    rewards_a = np.hstack([np.ones(len(actions_a))*positive_reward, np.ones(len(wrong_action_a))*negative_reward])


    actions_b = np.random.uniform(
        low=actions_b_range[0],
        high=actions_b_range[1],
        size=(int(sum(context_ids == 1) * (1 - percentage_of_wrong_actions)),),
    )
    wrong_action_b = np.random.uniform(
        low=actions_b_wrong_range[0],
        high=actions_b_wrong_range[1],
        size=(int(sum(context_ids == 1) * percentage_of_wrong_actions),),
    )
    played_actions_b = np.hstack([actions_b, wrong_action_b])
    rewards_b = np.hstack([np.ones(len(actions_b))*positive_reward, np.ones(len(wrong_action_b))*negative_reward])

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

    # plt.figure(figsize=(16, 9))
    #
    # x = np.arange(len(played_actions))
    # plt.title("Actions")
    # plt.plot(
    #     x[reward == negative_reward],
    #     played_actions[reward == negative_reward],
    #     label="played",
    #     linestyle="",
    #     marker="^",
    #     color="r",
    # )
    # plt.plot(
    #     x[reward == positive_reward],
    #     played_actions[reward == positive_reward],
    #     label="played",
    #     linestyle="",
    #     marker="*",
    #     color="b",
    # )
    #
    # plt.show()

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


    if True:
        num_epochs = 256
        sample_size = 256
        lr = 0.2

        steps = 100
        step_size = 0.2

        model = EBMModel(in_features_size=3)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )  # Exponential decay over epochs
        metric_watcher = defaultdict(list)

        xp = np.hstack(
            [np.vstack(past_contexts), np.vstack(past_rewards)]
        )
        xp = torch.tensor(xp)
        yp = torch.FloatTensor(past_actions)

        xp_eval = np.hstack(
            [
                np.vstack(eval_past_contexts),
                np.vstack(eval_past_rewards),
            ]
        )
        xp_eval = torch.tensor(xp_eval)
        yp_eval = torch.FloatTensor(eval_past_actions)

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(xp.size()[0])
            xp = xp[permutation, :]
            yp = yp[permutation]
            running_loss = []

            for xp_batch, yp_batch in zip(xp.split(sample_size), yp.split(sample_size)):
                optimizer.zero_grad()
                y = model(xp_batch,yp_batch)
                loss = y.mean()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())

            scheduler.step()

            # Evaluation
            model.eval()
            eval_running_loss = []
            for xp_batch, yp_batch in zip(xp_eval.split(sample_size), yp_eval.split(sample_size)):
                y = model(xp_batch, yp_batch)
                loss = y.mean()
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

        # TEST
        xp_test = np.hstack(
            [np.vstack(test_past_contexts), np.vstack(test_past_rewards)]
        )
        xp_test = torch.tensor(xp_test)
        yp_test = torch.FloatTensor(test_past_actions)

        model.eval()
        test_running_loss = []

        for xp_batch, yp_batch in zip(xp_test.split(sample_size), yp_test.split(sample_size)):
            y = model(xp_batch, yp_batch)
            loss = y.mean()
            test_running_loss.append(loss.item())

        print(
            f"Test loss {np.mean(test_running_loss):0.4f}"
        )

        # # Plot some shit
        # #  Action A
        # taken_context = test_past_contexts[test_context_ids==0][0]
        # taken_context_plus_reward = torch.FloatTensor(np.hstack((taken_context, [positive_reward]))).repeat(128, 1)
        # actions = np.linspace(plotting_range[0],plotting_range[1], 128)
        # energy = model(taken_context_plus_reward,torch.FloatTensor(actions))
        # plt.plot(actions, energy.detach().numpy())
        # plt.title("Energy function for positive reward action A")
        # plt.ylabel('Action')
        # plt.axvline(x=actions_a_range[0])
        # plt.axvline(x=actions_a_range[1])
        # plt.show()
        #
        # #  Action B
        # taken_context = test_past_contexts[test_context_ids==1][0]
        # taken_context_plus_reward = torch.FloatTensor(np.hstack((taken_context,[positive_reward]))).repeat(128, 1)
        # actions = np.linspace(plotting_range[0],plotting_range[1], 128)
        # energy = model(taken_context_plus_reward,torch.FloatTensor(actions))
        # plt.plot(actions,energy.detach().numpy())
        # plt.title("Energy function for positive reward action B")
        # plt.ylabel('Action')
        # plt.axvline(x=actions_b_range[0])
        # plt.axvline(x=actions_b_range[1])
        # plt.show()
        #
        #
        # # Test
        #
        # taken_context = test_past_contexts[test_context_ids==0][0]
        # taken_context_plus_reward = torch.FloatTensor(np.hstack((taken_context,[positive_reward])))
        #
        # action_to_play = torch.rand(1)
        #
        # model.eval()
        # for p in model.parameters():
        #     p.requires_grad = False
        # action_to_play.requires_grad = True
        #
        # # Enable gradient calculation if not already the case
        # had_gradients_enabled = torch.is_grad_enabled()
        # torch.set_grad_enabled(True)
        #
        # # We use a buffer tensor in which we generate noise each loop iteration.
        # # More efficient than creating a new tensor every iteration.
        # noise = torch.randn(action_to_play.shape, device=action_to_play.device)
        #
        # # List for storing generations at each step (for later analysis)
        # action_per_step = []
        # # Loop over K (steps)
        # for _ in range(steps):
        #     # Part 1: Add noise to the input.
        #     noise.normal_(0, 0.005)
        #     action_to_play.data.add_(noise.data)
        #
        #     # Part 2: calculate gradients for the current input.
        #     state = -model(taken_context_plus_reward,  action_to_play)
        #     state.sum().backward()
        #
        #     # Apply gradients to our current samples
        #     action_to_play.data.add_(step_size * action_to_play.grad.data)
        #     action_to_play.grad.detach_()
        #     action_to_play.grad.zero_()
        #     # inp_features.data.clamp_(min=0, max=10.0)
        #
        #     action_per_step.append(action_to_play.clone().detach())
        #
        # plt.plot(np.arange(len(action_per_step)), action_per_step, color="b")
        # plt.plot(0, action_per_step[0], label="initial_state", marker="o")
        # plt.plot(
        #     len(action_per_step) - 1, action_per_step[-1], label="final_state", marker="o"
        # )
        #
        # plt.axhline(actions_a_range[1], label="maximum", color="b")
        # plt.axhline(actions_a_range[0], label="minimun", color="r")
        # plt.title(f"Optimal action investigation")
        # plt.xlabel("Steps")
        # plt.ylabel("Action")
        # plt.legend()
        # plt.show()


        # Performances of guessing the correct answer:
        # TEST
        positive_reward_contexts = np.hstack(
            [np.vstack(test_past_contexts), np.ones(len(test_past_contexts)).reshape(-1,1)]
        )

        xp_test = torch.tensor(xp_test)
        yp_test = torch.FloatTensor(test_past_actions)

        final_results = []
        for positive_reward_context,context_id in zip(positive_reward_contexts,test_context_ids):
            positive_reward_context = torch.FloatTensor(positive_reward_context)
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
            noise = torch.randn(action_to_play.shape, device=action_to_play.device)

            # List for storing generations at each step (for later analysis)
            action_per_step = []
            # Loop over K (steps)
            for _ in range(steps):
                # Part 1: Add noise to the input.
                noise.normal_(0, 0.005)
                action_to_play.data.add_(noise.data)

                # Part 2: calculate gradients for the current input.
                state = -model(positive_reward_context, action_to_play)
                state.sum().backward()

                # Apply gradients to our current samples
                action_to_play.data.add_(step_size * action_to_play.grad.data)
                action_to_play.grad.detach_()
                action_to_play.grad.zero_()

            final_action = action_to_play.clone().detach().item()



            if context_id == 0:
                final_results.append(actions_a_range[0]<=final_action<=actions_a_range[1])
            else:
                final_results.append(actions_b_range[0] <= final_action <= actions_b_range[1])

        print(f'Accuracy {np.mean(final_results):0.4f}')
