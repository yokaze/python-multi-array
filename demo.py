import demo_module
import multi_array as ma
import numpy as np

x = ma.make(100, ma.float32)
x.set(np.random.rand(100))

# calculate average using a custom C++ module
print demo_module.average(x)
