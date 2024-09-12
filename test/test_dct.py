import numpy as np
def test(x0):
    x0 = np.transpose(x0, (2, 0, 1))
    return x0

image = np.random.rand(32, 32, 3)
print(image[0, 0 ,0 ])
test(image)
print(image[0, 0, 0])