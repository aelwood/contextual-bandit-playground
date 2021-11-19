from __future__ import annotations
import pickle
import typing
import matplotlib.pyplot as plt
import mlflow
import numpy as np

if typing.TYPE_CHECKING:
    from typing import Any, Iterable, NamedTuple, Optional, Sequence, Type, Union


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


class Evaluator:
    def __init__(
        self,
        run_name=None,
        save_data=False,
        saving_dir="./data/",
        plot_data=True,
        use_mlflow=None,
        policy=None,
        environment=None,
    ):
        self.oracle = {"actions": [], "rewards": [], "stochastic_reward": []}
        self.policy = {"actions": [], "rewards": [], "stochastic_reward": []}

        if run_name:
            self.run_name = run_name
        else:
            self.run_name = str(np.datetime64("now")).replace(":", "-")

        self.save_data = save_data
        self.plot_data = plot_data
        self.saving_dir = saving_dir
        self.use_mlflow = use_mlflow
        self.active_policy = policy
        self.active_environment = environment

        if self.use_mlflow:
            self.cumulative_stochastic_reward_policy = 0.0
            self.cumulative_stochastic_reward_oracle = 0.0
            self.cumulative_expected_reward_policy = 0.0
            self.cumulative_expected_reward_oracle = 0.0
            self.step = 0

            mlflow.start_run(run_name=self.run_name)

            # logs params shall call policy and env params

            mlflow.log_param("env_name", self.active_environment.name)
            mlflow.log_param("policy_name", self.active_policy.name)

            params = self.active_policy.get_params()
            if params:
                for k, v in params.items():
                    mlflow.log_param(k, v)

            params = self.active_environment.get_params()
            if params:
                for k, v in params.items():
                    mlflow.log_param(k, v)

    def end(self):
        if self.use_mlflow:
            mlflow.end_run()
        self.plot(self.plot_data)
        if self.save_data:
            self.save()

    def save(self):
        file_to_save = {"oracle": self.oracle, "policy": self.policy}
        path = self.saving_dir + "dump/" + self.run_name
        save_obj(file_to_save, path)
        print(f"File saved to {path}")

    # (a, r,s_r, optimal_a, optimal_r,stochastic_r)
    def notify(
        self,
        played_action,
        obtained_reward,
        obtained_stochastic_reward,
        optimal_action,
        optimal_reward,
        stochastic_reward,
    ):
        self.policy["actions"].append(played_action)
        self.policy["rewards"].append(obtained_reward)
        self.policy["stochastic_reward"].append(obtained_stochastic_reward)
        self.oracle["actions"].append(optimal_action)
        self.oracle["rewards"].append(optimal_reward)
        self.oracle["stochastic_reward"].append(stochastic_reward)

        if self.use_mlflow:
            self.cumulative_stochastic_reward_policy += obtained_stochastic_reward
            self.cumulative_stochastic_reward_oracle += stochastic_reward
            self.cumulative_expected_reward_policy += obtained_reward
            self.cumulative_expected_reward_oracle += optimal_reward
            stochastic_regret = (
                self.cumulative_stochastic_reward_oracle
                - self.cumulative_stochastic_reward_policy
            )
            expected_regret = (
                self.cumulative_expected_reward_oracle
                - self.cumulative_expected_reward_policy
            )

            mlflow.log_metrics(
                {
                    "stochastic_regret": stochastic_regret,
                    "expected_regret": expected_regret,
                    "played_action": played_action,
                    "obtained_reward": obtained_reward,
                    "obtained_stochastic_reward": obtained_stochastic_reward * 1,
                    "optimal_action": optimal_action,
                    "optimal_reward": optimal_reward,
                    "stochastic_reward": stochastic_reward * 1,
                    "cumulative_stochastic_reward_policy": self.cumulative_stochastic_reward_policy,
                    "cumulative_stochastic_reward_oracle": self.cumulative_stochastic_reward_oracle,
                    "cumulative_expected_reward_policy": self.cumulative_expected_reward_policy,
                    "cumulative_expected_reward_oracle": self.cumulative_expected_reward_oracle,
                },
                step=self.step,
            )
            self.step += 1

    def get_stats(self):
        return f"Average oracle reward {np.mean(self.oracle['rewards']):2.4f} vs policy {np.mean(self.policy['rewards']):2.4f}"

    def plot(self, display):

        plt.figure(figsize=(16, 9))

        x = np.arange(len(self.policy["actions"]))

        ax1 = plt.subplot(211)

        ax1.set_title("Played actions")
        ax1.plot(x, self.policy["actions"], label="policy", linestyle="", marker="^")
        ax1.plot(x, self.oracle["actions"], label="oracle", linestyle="", marker="x")

        ax2 = plt.subplot(223)
        ax2.set_title("Cumulative Rewards")
        ax2.plot(
            x, np.cumsum(self.policy["rewards"]), label="policy(expected)", alpha=0.6
        )
        ax2.plot(
            x, np.cumsum(self.oracle["rewards"]), label="oracle(expected)", alpha=0.6
        )
        ax2.plot(
            x,
            np.cumsum(self.policy["stochastic_reward"]),
            label="policy(stochastic)",
            alpha=0.6,
        )
        ax2.plot(
            x,
            np.cumsum(self.oracle["stochastic_reward"]),
            label="oracle(stochastic)",
            alpha=0.6,
        )
        ax2.legend()

        ax3 = plt.subplot(224)
        ax3.set_title("Cumulative Regrets")
        ax3.plot(
            x,
            np.cumsum(self.oracle["stochastic_reward"])
            - np.cumsum(self.policy["stochastic_reward"]),
            label="stochastic_Regrets",
            lw=4,
            alpha=0.8,
        )
        ax3.plot(
            x,
            np.cumsum(self.oracle["rewards"]) - np.cumsum(self.policy["rewards"]),
            label="Expected_Regrets",
            lw=4,
            alpha=0.8,
        )

        ax3.legend()
        ax1.legend()
        if display:
            plt.show()
        else:
            path = self.saving_dir + "plots/" + self.run_name + ".png"
            plt.savefig(path)
