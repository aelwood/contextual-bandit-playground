from policies import PolicyABC

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# class EnergyBasedModelMseNaive(nn.Module):
#     def __init__(self, in_features_size=32, hidden_feature_size=[32, 16, 8]):
#         super(EnergyBasedModelMseNaive, self).__init__()
#         feat_sizes = [in_features_size] + hidden_feature_size
#
#         cnn_layers = nn.ModuleList([
#             torch.nn.Linear(in_features_size, 200),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(200, 100),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(100, 1),
#         ])
#
#         self.layers = cnn_layers
#
#     def reinitialize_weights(self):
#         for layer in self.layers:
#             if isinstance(layer, nn.Linear):
#                 torch.nn.init.xavier_uniform(layer.weight)
#                 layer.bias.data.fill_(0.01)
#
#     def forward(self, x, y):
#         """
#         :param x: [context,reward]
#         :param y: [action]
#         :return: energy
#         """
#         y = y
#         x = x.float()
#         for i in range(len(self.layers)):
#             x = self.layers[i](x)
#         x = x.squeeze(dim=-1)
#         energy = 1 / 2 * torch.pow(x - y, 2)
#         # energy = torch.abs(x-y)
#         return energy
#
#
class EBMModelImplicitRegression(nn.Module):
    def __init__(
        self, in_features_size=32, output_quadratic: bool = True, use_dropout=True
    ):
        super(EBMModelImplicitRegression, self).__init__()
        if use_dropout:
            self.g_1 = nn.ModuleList(
                [
                    torch.nn.Linear(in_features_size, 256),
                    torch.nn.Sigmoid(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(256, 256),
                    torch.nn.Sigmoid(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(256, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(128, 1),
                ]
            )

            self.g_2 = nn.ModuleList(
                [
                    torch.nn.Linear(1, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(128, 1),
                ]
            )
        else:
            self.g_1 = nn.ModuleList(
                [
                    torch.nn.Linear(in_features_size, 256),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(256, 256),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(256, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(128, 1),
                ]
            )

            self.g_2 = nn.ModuleList(
                [
                    torch.nn.Linear(1, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(128, 1),
                ]
            )
        self.output_quadratic = output_quadratic

    def reinitialize_weights(self, mode: str):
        print(f'Reinitializing model with {mode}')
        if mode is None or mode == "":
            return
        if mode == "custom":
            for layer in self.g_1:
                if isinstance(layer, nn.Linear):
                    stdv = 1.0 / math.sqrt(layer.weight.size(1))
                    layer.weight.data.uniform_(-stdv, stdv)
                    if layer.bias is not None:
                        layer.bias.data.uniform_(-stdv, stdv)
            for layer in self.g_2:
                if isinstance(layer, nn.Linear):
                    stdv = 1.0 / math.sqrt(layer.weight.size(1))
                    layer.weight.data.uniform_(-stdv, stdv)
                    if layer.bias is not None:
                        layer.bias.data.uniform_(-stdv, stdv)
        if mode == "xavier":
            for layer in self.g_1:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    layer.bias.data.fill_(0.01)
            for layer in self.g_2:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    layer.bias.data.fill_(0.01)

        if mode == "xavier_1":
            for layer in self.g_1:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    layer.bias.data.fill_(0.0)
            for layer in self.g_2:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    layer.bias.data.fill_(0.0)

        if mode == "xavier_2":
            for layer in self.g_1:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    layer.bias.data.fill_(0.1)
            for layer in self.g_2:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)
                    layer.bias.data.fill_(0.1)

        if mode == "kaiming":
            for layer in self.g_1:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0.0)
            for layer in self.g_2:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0.0)

        if mode == "kaiming2":
            for layer in self.g_1:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0.01)
            for layer in self.g_2:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                    nn.init.constant_(layer.bias.data, 0.01)

        print("Done!")

    def forward(self, x, y, yo=False):
        """
        :param x: [context,reward]
        :param y: [action]
        :return: energy
        """
        x = x.float()
        for i in range(len(self.g_1)):
            x = self.g_1[i](x)
        x = x.squeeze(dim=-1)

        yn = y.float()
        yn = yn.unsqueeze(1)
        for i in range(len(self.g_2)):
            yn = self.g_2[i](yn)
        yn = yn.squeeze(dim=-1)
        if yo:
            for gamma, (_y, _yn) in enumerate(zip(y, yn)):
                print(f"{_y} + {_yn} = {_y + _yn}")
                if gamma == 2:
                    break
        y = y + yn
        y = y.squeeze(dim=-1)

        # energy = torch.abs(x-y)
        # energy = x-y
        if self.output_quadratic:
            energy = 1 / 2 * torch.pow(x - y, 2)
        else:
            energy = torch.abs(x - y)
        # energy = torch.log(1 + torch.exp(x-y))

        return energy


class EBMPolicy(PolicyABC):
    def set_model(self):
        self.ebm_estimator = self.ebm_estimator_class(
            in_features_size=self.adjusted_feat_size,
            output_quadratic=self.output_quadratic,
            use_dropout=self.use_dropout,
        )

        self.ebm_estimator.reinitialize_weights(self.init_techq)

        self.optimizer = optim.Adam(self.ebm_estimator.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.schedule_step_size, gamma=0.1
        )

    def __init__(
        self,
        name="EnergyBasedModel",
        feature_size=2,
        ebm_estimator_class=EBMModelImplicitRegression,
        lr=0.05,
        sample_size=128,
        warm_up=64,
        num_epochs=50,
        loss_function_type: str = "log",
        device=torch.device("cpu"),
        schedule_step_size: int = 100,
        output_quadratic: bool = True,
        use_dropout: bool = True,
        alpha: float = 10.0,
        sw=0,
        init_techq=None,
    ):
        self.past_rewards = []
        self.past_actions = []

        self.past_contexts = []
        self.warm_up = warm_up
        self.sample_size = sample_size
        self.last_training_idx = 0
        self.num_epochs = num_epochs
        assert loss_function_type in ["log", "mce", "log2"]
        self.loss_function_type = loss_function_type
        self.feature_size = feature_size
        self.adjusted_feat_size = feature_size
        self.alpha = alpha
        self.first_train = True
        self.first_train_retry = 10
        self.max_retry = 55

        self.name = name

        self.positive_reward = 1
        self.negative_reward = -1
        self.ebm_estimator_class = ebm_estimator_class

        self.output_quadratic = output_quadratic
        self.use_dropout = use_dropout
        self.init_techq = init_techq
        self.schedule_step_size=schedule_step_size
        self.lr = lr
        self.set_model()

        sw *= -1 if sw > 0 else 1
        self.sw = sw



    def __copy__(self):
        return self.__class__(
            name=self.name,
            feature_size=self.feature_size,
            ebm_estimator_class=self.ebm_estimator_class,
            lr=self.lr,
            sample_size=self.sample_size,
            warm_up=self.warm_up,
            num_epochs=self.num_epochs,
            loss_function_type=self.loss_function_type,
            device=torch.device("cpu"),
            sw=self.sw,
        )

    def notify_event(self, context, action, stochastic_reward):
        self.past_contexts.append(context)
        if int(stochastic_reward) == 0:
            self.past_rewards.append(-1)
        else:
            self.past_rewards.append(int(stochastic_reward))
        self.past_actions.append(action)

    def train(self):
        train_output = self._train()

        if train_output:
            # if len(self.past_rewards)<-1:
            if self.first_train:
                means = []
                models_params = []

                positive_samples_ids = np.argwhere(np.array(self.past_rewards[self.sw:]) == 1)
                contexts = np.array(self.past_contexts[self.sw:])[positive_samples_ids].squeeze()
                actions = np.array(self.past_actions[self.sw:])[positive_samples_ids].squeeze()
                print('First train, looking for the best model')
                rtr = 0
                while self.first_train_retry > rtr or min(means) > 0.6:
                    rtr += 1
                    print(f'{rtr} of {self.first_train_retry}')

                    played_actions = np.array([self.get_action(context) for context in contexts])

                    means.append(np.mean(np.abs(actions-played_actions)))

                    models_params.append(self.ebm_estimator.state_dict())

                    # a1 = 1
                    # a2 = 4
                    # c1 = contexts[np.argmin(np.abs(actions - 1))]
                    # c2 = contexts[np.argmin(np.abs(actions - 4))]
                    #
                    # number_of_points = 100
                    # for action, context in zip([a1, a2], [c1, c2]):
                    #     context_plus_reward = torch.FloatTensor(context).repeat(number_of_points, 1)
                    #     actionss = np.linspace(0, 5, number_of_points)
                    #     energy = self.ebm_estimator(context_plus_reward, torch.FloatTensor(actionss))
                    #     plt.plot(actionss, energy.detach().numpy(), label=f'Action {action}')
                    # plt.title(f'mean difference: {np.mean(np.abs(actions-played_actions))}')
                    # plt.legend()
                    # plt.show()
                    self.set_model()
                    self._train()
                    if self.max_retry< rtr:
                        break

                print(f'Mean differences {means}')
                best_param = models_params[np.argmin(means)]
                self.ebm_estimator.load_state_dict(best_param)
                self.first_train = False
                # played_actions = np.array([self.get_action(context) for context in contexts])
                # print(f"{means.append(np.mean(np.abs(actions - played_actions)))}")





    def _train(self):
        sample_size = self.sample_size
        if self.warm_up > len(self.past_contexts):
            return None

        # self.ebm_estimator.reinitialize_weights()

        context_to_train = self.past_contexts[self.sw :]
        actions_to_train = self.past_actions[self.sw :]
        rewards_to_train = self.past_rewards[self.sw :]

        self.ebm_estimator.train()

        for p in self.ebm_estimator.parameters():
            p.requires_grad = True

        xp = np.hstack([np.vstack(context_to_train), np.vstack(rewards_to_train)])
        xp = torch.tensor(xp)
        yp = torch.FloatTensor(actions_to_train)

        for epoch in range(self.num_epochs):

            permutation = torch.randperm(xp.size()[0])

            xp = xp[permutation, :]
            yp = yp[permutation]

            running_loss = []
            for i, (xp_batch, yp_batch) in enumerate(
                zip(xp.split(sample_size), yp.split(sample_size))
            ):
                # xp_batch.requires_grad=True
                # yp_batch.requires_grad=True

                positive_reward_mask = xp_batch[:, -1] == self.positive_reward
                negative_reward_mask = xp_batch[:, -1] == self.negative_reward

                positive_xp_batch = xp_batch[positive_reward_mask, :-1]
                positive_yp_batch = yp_batch[positive_reward_mask]
                negative_xp_batch = xp_batch[negative_reward_mask, :-1]
                negative_yp_batch = yp_batch[negative_reward_mask]

                # TODO discuss: we're throwing away negative rewards here?
                #  thi s is a big possible problem when there are few positive examples
                #  try upsampling?
                #  Another thing to try is just collect loads of random samples and see if it works
                #  ie try with a bigger warmup

                min_len = min(len(positive_xp_batch), len(negative_xp_batch))

                # print(min_len)
                if min_len == 0:
                    continue  # TODO check

                positive_xp_batch = positive_xp_batch[:min_len, :]
                positive_yp_batch = positive_yp_batch[:min_len]
                negative_xp_batch = negative_xp_batch[:min_len, :]
                negative_yp_batch = negative_yp_batch[:min_len]

                self.optimizer.zero_grad()

                y_pos = self.ebm_estimator(positive_xp_batch, positive_yp_batch)
                y_neg = self.ebm_estimator(negative_xp_batch, negative_yp_batch)

                if self.loss_function_type == "log":
                    loss = torch.log(1 + torch.exp(y_pos - y_neg)).mean()
                elif self.loss_function_type == "log2":
                    loss = torch.log(1 + torch.exp(10 * y_pos - 10 * y_neg)).mean()
                elif (
                    self.loss_function_type == "mce"
                ):  # todo understand why this doesn't work
                    # todo it seems like the model output becomes exponential
                    loss = torch.pow((1 + torch.exp(-(y_pos - y_neg))), -1).mean()
                else:
                    raise ValueError

                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())

            if epoch % 30 == 0:
                print(f"Epoch {epoch} avg training loss {np.mean(running_loss):0.4f}")
            self.scheduler.step()
        self.ebm_estimator.eval()
        return True

    def get_action(self, context):
        if self.warm_up >= len(self.past_contexts):
            return np.random.rand() * 5

        steps = 100
        alpha = self.alpha
        step_size = 0.2  # TODO discuss
        # taken_context_plus_reward = torch.FloatTensor(np.hstack((context, [1])))
        context = torch.FloatTensor(context)

        action_to_play = torch.rand(1) * 5

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
        action_per_step = []
        for _ in range(steps):
            # Part 1: Add noise to the input.
            action_per_step.append(action_to_play.clone().detach())

            noise.normal_(0, 0.005)
            action_to_play.data.add_(noise.data)

            # Part 2: calculate gradients for the current input.
            state = -self.ebm_estimator(context, action_to_play) / alpha
            state.sum().backward()

            # Apply gradients to our current samples
            action_to_play.data.add_(step_size * action_to_play.grad.data)
            action_to_play.grad.detach_()
            action_to_play.grad.zero_()
            # inp_features.data.clamp_(min=0, max=10.0)

        return float(action_to_play.clone().detach().item())

    def get_params(self):
        pass


def plot_energy(context, model):
    actions = np.linspace(-10, 10, 100)
    energy = model(context, torch.FloatTensor(actions))
    plt.plot(actions, energy.detach().numpy())
    plt.show()


def plot_sampling(action_per_step, _range):
    plt.plot(np.arange(len(action_per_step)), action_per_step, color="b")
    plt.plot(0, action_per_step[0], label="initial_state", marker="o")
    plt.plot(
        len(action_per_step) - 1, action_per_step[-1], label="final_state", marker="o"
    )
    plt.axhline(_range[1], label="maximum", color="b")
    plt.axhline(_range[0], label="minimun", color="r")
    plt.title(f"Optimal action investigation")
    plt.xlabel("Steps")
    plt.ylabel("Action")
    plt.legend()
    plt.show()
