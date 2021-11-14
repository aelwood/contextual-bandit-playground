from __future__ import annotations

import abc
import numpy as np
import typing

from abc import ABC

from sklearn.datasets import make_blobs
from scipy.stats import norm
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    for context_id in range(3):
        mu = context_id * 4 + 1
        sigma = 0.6
        x = np.linspace(mu - 2.5, mu + 2.5, 100)
        plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), label=f"context {context_id}")
    plt.legend()
    plt.show()
