# Test for https://github.com/parrt/tensor-sensor/issues/18
# Nested clarify's and all catch exception

import tsensor
import numpy as np

def f():
    np.ones(1) @ np.ones(2)

def A():
    with tsensor.clarify():
        f()

def B():
    with tsensor.clarify():
        A()

B()