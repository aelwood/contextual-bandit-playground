import torch
from matplotlib import pyplot as plt
import numpy as np
import datetime

from collections import defaultdict
from environments import CirclesSyntheticEnvironment
from policies_energy_based import EBMPolicy
from LSUV import LSUVinit


# ## setting the random seeds, for easy testing and developments
# random_seed = 42
# random.seed(random_seed)
# np.random.seed(random_seed)
from pathlib import Path

if __name__ == "__main__":
    saving_dir = Path("/Users/mleonardi/Projects/research/contextual-bandit-playground/data/")

    pretrain_time = 1_000
    number_of_observations = 1_750
    train_every = 100

    lr = 0.005
    num_epochs = 150
    loss_function_type = "log"
    sample_size = 256
    output_quadratic = False
    alpha = 10
    init_techq = 'xavier'
    use_dropout=True

    for init_techq in ['',]:
        if output_quadratic:
            var = 'QUAD'
        else:
            var = 'LIN'
        exp_cluster = f"FINALONESUPER-{train_every:,}_initializer-{init_techq}_LR-{str(lr).split('.')[-1]}_EPOCH-{num_epochs}_LFT-{loss_function_type}_SS-{sample_size}_ALPHA-{str(alpha).split('.')[-1]}_{var}"

        if not use_dropout:
            exp_cluster += '_no-dropout'

        for _ in range(5):
            environment = CirclesSyntheticEnvironment(
                number_of_different_context=2,
                n_context_features=2,
                number_of_observations=number_of_observations,
                time_perturbation_function=lambda time, mu: mu,
                action_offset=3,
                circle_factor=4.,
                name="2c_4_circ",
            )

            policy = EBMPolicy(
                name=f'TEST',
                lr=lr,
                warm_up=pretrain_time,
                num_epochs=num_epochs,
                loss_function_type=loss_function_type,
                sample_size=sample_size,
                output_quadratic=output_quadratic,
                alpha=alpha,
                feature_size=2,
                init_techq=init_techq,
             )

            c1 = None
            c2 = None
            a1 = None
            a2 = None

            stats_dict = defaultdict(list)
            for i, c in enumerate(environment.generate_contexts()):
                a = policy.get_action(c)
                r, s_r = environment.get_reward(a, c)
                policy.notify_event(c, a, s_r)

                optimal_r, optimal_a, stochastic_r = environment.get_best_reward_action(c)

                stats_dict['a'].append(a)
                stats_dict['r'].append(r)
                stats_dict['s_r'].append(s_r)
                stats_dict['optimal_a'].append(optimal_a)
                stats_dict['optimal_r'].append(optimal_r)
                stats_dict['stochastic_r'].append(stochastic_r)

                if c1 is None:
                    c1 = c
                    a1 = optimal_a

                if c2 is None and optimal_a!= a1:
                    c2 = c
                    a2 = optimal_a

                if i % train_every == 0:
                    print(f'Observed {i} items over {number_of_observations}.')
                    policy.train()


            cumulative_stochastic_reward_policy = np.cumsum(stats_dict['s_r'])
            cumulative_stochastic_reward_oracle = np.cumsum(stats_dict['stochastic_r'])
            stochastic_regret = cumulative_stochastic_reward_oracle - cumulative_stochastic_reward_policy

            plt.figure(figsize=(16, 9))
            x = np.arange(len(stats_dict['a']))

            ax1 = plt.subplot(221)
            ax1.set_title("Played actions")
            ax1.plot(x, stats_dict['a'], label="policy", linestyle="", marker="^")
            ax1.plot(x, stats_dict['optimal_a'], label="oracle", linestyle="", marker="x")


            ####
            ax4 = plt.subplot(222)
            model = policy.ebm_estimator
            number_of_points = 100
            for action, context in zip([a1,a2],[c1, c2]):
                context_plus_reward = torch.FloatTensor(context).repeat(number_of_points, 1)
                actions = np.linspace(0,5, number_of_points)
                energy = model(context_plus_reward, torch.FloatTensor(actions))
                ax4.plot(actions, energy.detach().numpy(), label=f'Action {action}')

            ax4.set_title("Energy function")
            ax4.axvline(x=1)
            ax4.axvline(x=4)




            ax3 = plt.subplot(224)
            ax3.set_title("Cumulative Regrets")
            ax3.plot(
                x,
                stochastic_regret,
                label="stochastic_regret",
                lw=4,
                alpha=0.8,
            )
            ax3.legend()
            ax4.legend()
            ax1.legend()

            ax2 = plt.subplot(223)

            text = (f'pretrain_time:{pretrain_time}\n'+
                    f'number_of_observations: {number_of_observations}\n'+
                    f'train_every: {train_every}\n'+
                    f'lr: {lr}\n'+
                    f'num_epochs: {num_epochs}\n'+
                    f'loss_function_type: {loss_function_type}\n'+
                    f'sample_size: {sample_size}\n'+
                    f'output_quadratic: {output_quadratic}\n'+
                    f'alpha: {alpha}\n\n' +
                    f'STOCHASTIC REGRET: {stochastic_regret[-1]:,}\n'

                    )

            plt.axis([0, 10, 0, 10])

            plt.text(2, 8, text, fontsize=12, ha='left',
                     va='top', wrap=True)
            plt.axis('off')

            path = (saving_dir / "SUPER_EXPS" /exp_cluster/ datetime.datetime.now().strftime('%Y%m%d%H%M%S')).with_suffix(".png")
            path.parents[0].mkdir(exist_ok=True)
            plt.savefig(path)
            plt.close()





