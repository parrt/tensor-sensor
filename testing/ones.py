# Regression test for https://github.com/parrt/tensor-sensor/issues/16
# Should not throw exception or error, just show equation.
import numpy as np
import tsensor

print(tsensor.__version__)
with tsensor.explain():
   a = np.ones(3)
