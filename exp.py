import numpy as np
samp1 = np.random.power(a=0.83, size=10)
samp2 = np.random.power(a=0.83, size=10)
print(samp1)
print(samp2)
print(np.hstack([samp1, samp2]))