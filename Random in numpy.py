import numpy as np

print(np.random.randn(10))
np.random.seed(2)
print(np.random.randn(1,5))
print(np.random.randint(0,10,(3,3)))
print(np.random.random((3,3)))