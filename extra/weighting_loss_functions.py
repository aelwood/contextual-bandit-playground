from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    past_actions = np.random.random(800)

    min_value = 0.1

    # # Naive:
    # memory = 200
    # memory = min(memory,len(past_actions))
    # weights = np.array([min_value] * len(past_actions))
    # starting_point = len(past_actions)-memory
    # q = (1-min_value)/memory
    # weights[-memory:] += np.arange(1, memory+1) * q
    #
    # plt.plot(np.arange(len(past_actions)),weights)
    # plt.title(f'Naive scaling: memory={memory}')
    # plt.show()

    # Naive 2:
    # short_term_mem = 100  # all 1
    # long_term_mem = 100  # all 0
    #
    # short_term_mem = min(short_term_mem,len(past_actions))
    # long_term_mem = max(min(long_term_mem,len(past_actions)-short_term_mem),0)
    #
    # memory = short_term_mem + long_term_mem
    # weights = np.array([min_value] * len(past_actions))
    #
    # starting_point = len(past_actions)-memory
    # if long_term_mem>0:
    #     q = (1-min_value)/long_term_mem
    #     weights[-memory:-short_term_mem] += np.arange(1, long_term_mem+1) * q
    #
    # weights[-short_term_mem:] = 1.
    #
    # plt.plot(np.arange(len(past_actions)),weights)
    # plt.title(f'Naive scaling: memory={memory}')
    # plt.show()

    # Sigmoid:
    # TODO
    short_term_mem = 200 # 300
    long_term_mem = 100 # 300
    short_term_mem = min(short_term_mem, len(past_actions))
    long_term_mem = max(min(long_term_mem, len(past_actions) - short_term_mem), 0)

    memory = short_term_mem + long_term_mem

    starting_point = len(past_actions) - memory
    if long_term_mem > 0:
        q = 1 / long_term_mem
        a = len(past_actions) - long_term_mem - short_term_mem
        b = len(past_actions) - short_term_mem
        custom_sigmoid = lambda x: 1 / (1 + np.exp((q * 4) * (-x + a + (b - a) / 2)))

        weights = custom_sigmoid(np.arange(1, len(past_actions) + 1))
    else:
        weights = np.array([1.0] * len(past_actions))

    plt.plot(np.arange(len(past_actions)), weights)

    plt.xticks(
        np.arange(len(past_actions), -1, -100)[::-1],
        np.arange(len(past_actions), -1, -100),
    )

    plt.axvline(len(past_actions) - short_term_mem, label="short")
    plt.axvline(len(past_actions) - long_term_mem - short_term_mem, label="long")

    plt.title(f"Sigmoid scaling: memory={memory}")
    plt.show()

    #
    #
    # # Playng around with sigmoid
    # a = 80
    # b = 150
    # q = 1 / (b - a)
    #
    # sigmoid = lambda x: 1 / (1 + np.exp((-x + a + (b - a) / 2)))  # + long_-short_
    #
    # plt.plot(np.arange(-2, 200, .1), sigmoid(np.arange(-2, 200, .1)), label='plain')
    # plt.plot(np.arange(-2, 200, .1), custom_sigmoid(np.arange(-2, 200, .1)), label='custom')
    # plt.axvline(a + (b - a) / 2, 0, 1)
    #
    # plt.axvline(a, 0, 1)
    # plt.axvline(b, 0, 1)
    #
    # plt.legend()
    # plt.show()
    #
    #

    #####

    # # Playng around with sigmoid
    # a = 80
    # b = 150
    # q = 1 / (b - a)
    # custom_sigmoid = lambda x: 1/(1 + np.exp((q*4)*(-x + a + (b - a)/2)))
    # sigmoid = lambda x: 1/(1 + np.exp((-x + a + (b - a)/2))) #+ long_-short_
    #
    # plt.plot(np.arange(-2,200,.1),sigmoid(np.arange(-2,200,.1)),label='plain')
    # plt.plot(np.arange(-2, 200, .1), custom_sigmoid(np.arange(-2, 200, .1)), label='custom')
    # plt.axvline(a+(b - a)/2, 0, 1)
    #
    # plt.axvline(a, 0, 1)
    # plt.axvline(b, 0, 1)
    #
    # plt.legend()
    # plt.show()

    # memory = short_term_mem + long_term_mem
    # weights = np.array([min_value] * len(past_actions))
    #
    # starting_point = len(past_actions)-memory
    # if long_term_mem>0:
    #     q = (1-min_value)/long_term_mem
    #     weights[-memory:-short_term_mem] += np.arange(1, long_term_mem+1) * q
    #
    # weights[-short_term_mem:] = 1.
    #
    # plt.plot(np.arange(len(past_actions)),weights)
    # plt.title(f'Naive scaling: memory={memory}')
    # plt.show()
    #
    #
