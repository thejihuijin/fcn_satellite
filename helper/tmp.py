import numpy as np

A = np.ones((4,4))
print A*np.random.binomial(1, .8, A.shape)*255