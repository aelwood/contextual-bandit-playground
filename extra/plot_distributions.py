import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll, make_circles

action_offset = 3
number_of_different_context=2
fixed_variances=0.6
n_context_features=3
environment_best_action_offset=1
mul_factor = 1

context_reward_parameters = {
    context_id: {
        "mu": context_id * action_offset + environment_best_action_offset,
        "sigma": fixed_variances,
    }
    for context_id in range(number_of_different_context)
}


x_axis = np.arange(-1, 6, 0.1)

for context_id in range(number_of_different_context):
    plt.plot(x_axis, norm.pdf(x_axis, context_reward_parameters[context_id]['mu'], context_reward_parameters[context_id]['sigma']))
plt.title('Probability Density Function of the two reward functions')
plt.xlabel('Actions')
plt.ylabel('Probability')
plt.show()

context_vectors, context_ids = make_blobs(
            n_samples=10_000,
            n_features=2,
            centers=number_of_different_context,
            cluster_std=0.4,
            shuffle=True,
        )


for context_id in range(number_of_different_context):
    plt.scatter(context_vectors[context_ids==context_id, 0], context_vectors[context_ids==context_id, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title('Linearly separable context distribution')
plt.show()


circle_factor=4.

context_vectors, context_ids = make_circles(
    n_samples=10_000,
    shuffle=True,
)
context_vectors[context_ids == 0, 0] *= circle_factor
context_vectors[context_ids == 0, 1] *= circle_factor



for context_id in range(number_of_different_context):
    plt.scatter(context_vectors[context_ids==context_id, 0], context_vectors[context_ids==context_id, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title('Linearly separable context distribution')
plt.show()