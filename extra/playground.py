# from __future__ import annotations
#
# import abc
# import numpy as np
# import typing
#
# from abc import ABC
#
# import tensorflow as tf
# from sklearn.datasets import make_blobs
# from scipy.stats import norm
# from scipy.stats import norm
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# if __name__ == "__main__":
#
#     action_bounds = [1, 2]
#     reward_bounds = [0, 1]
#
#     # return (tf.sigmoid((action - self.action_bounds[1]) * 1000) * -999999 + 1) * \
#     #        (tf.sigmoid(-(action - self.action_bounds[0]) * 1000)*-999999 + 1) * \
#
#     # action limiting
#     # f =  lambda action: tf.sigmoid((action-action_bounds[0])*1000) # Passa alto
#     # f =  lambda action: (1-tf.sigmoid((action-action_bounds[1])*1000)) # Passa basso
#
#     # f = lambda action: tf.sigmoid((action-action_bounds[0])*1000) *\
#     #                    (1-tf.sigmoid((action-action_bounds[1])*1000)) # Passa banda
#
#     # new one
#     # f = lambda action: (tf.sigmoid((action-action_bounds[0])*100_000) *\
#     #                    (1-tf.sigmoid((action-action_bounds[1])*100_000)) -1) * 999999 + 1
#
#     # older version
#     # f = lambda action: (tf.sigmoid((action - action_bounds[1]) * 1000) * -999999 + 1) * \
#     #                     (tf.sigmoid(-(action - action_bounds[0]) * 1000)*-999999 + 1)
#
#     # lol they are the same
#
#     # f = lambda reward: reward
#     # f = lambda reward:  tf.math.minimum(reward_bounds[1],reward)
#     # f = lambda reward: np.min([reward_bounds[1], reward])
#
#     # f = lambda reward: tf.math.maximum(
#     #                reward_bounds[0],
#     #                tf.math.minimum(
#     #                    reward_bounds[1],
#     #                    reward
#     #                )
#     #            )
#
#     # f= lambda action: (tf.sigmoid((action - action_bounds[1]) * 1000) * -999999 + 1) * \
#     # (tf.sigmoid(-(action - action_bounds[0]) * 1000) * -999999 + 1) * \
#     # tf.math.maximum(
#     #     self.reward_bounds[0],
#     #     tf.math.minimum(
#     #         self.reward_bounds[1],
#     #         super(RewardLimiterMixin, self).predict_reward_maintaining_graph(
#     #             action, context
#     #         )
#     #     )
#     # )
#     f = (
#         lambda action, reward: (
#             tf.sigmoid((action - action_bounds[1]) * 1000) * -999999 + 1
#         )
#         + (tf.sigmoid(-(action - action_bounds[0]) * 1000) * -999999 + 1)
#         + tf.math.maximum(reward_bounds[0], tf.math.minimum(reward_bounds[1], reward))
#     )
#
#     # f = lambda reward:
#     #                tf.math.minimum(
#     #                    reward_bounds[1],
#     #                    reward
#     #                )
#     #            )
#
#     rg = np.arange(3, -2, -0.1)
#     plt.plot(
#         rg,
#         [
#             f(
#                 tf.convert_to_tensor(x, dtype=tf.float32),
#                 tf.convert_to_tensor(x, dtype=tf.float32),
#             )
#             for x in rg
#         ],
#         label="sigmoid",
#     )
#
#     plt.plot(rg, rg, label="input")
#
#     plt.legend()
#     plt.show()
#
# # clip reward to - 999999
# # FIXME - understand why this is wrong
# # return (tf.sigmoid((action - self.action_bounds[1]) * 1000) * -999999 + 1) * \
# #        (tf.sigmoid(-(action - self.action_bounds[0]) * 1000)*-999999 + 1) * \
#
# #        tf.math.maximum(
# #            self.reward_bounds[0],
# #            tf.math.minimum(
# #                self.reward_bounds[1],
# #                super(RewardLimiterMixin, self).predict_reward_maintaining_graph(
# #                    action, context
# #                )
# #            )
# #        )
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import matplotlib.animation as animation

fig = plt.figure()

ims = []
fig = plt.figure()
ax = fig.add_subplot(111)
for _ in range(100):
    im1, = ax.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)], animated=True)
    ims.append([im1])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
writergif = animation.PillowWriter(fps=30)
ani.save('movie.gif',writer=writergif)
# plt.show()