import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


mean = 4
sd = 0.1
mul_factor = 0.2
x_axis = np.arange(1, 8, .01)
plt.plot(x_axis, norm.pdf(x_axis, mean, sd)*mul_factor)
plt.show()


# for variance,mul in zip([60,50,40,30,20,10],[100,90,90,70,40,20])

# [0.6,0.5,0.4,0.3,0.2,0.1]
# [1,1,0.9,0.7,0.4,0.2]