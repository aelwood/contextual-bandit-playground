import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


mean = 400
sd = 60
mul_factor = 120
x_axis = np.arange(200, 600, 1)
plt.plot(x_axis, norm.pdf(x_axis, mean, sd)*mul_factor)
plt.show()