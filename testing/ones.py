import numpy as np
import tsensor

def foo():
   x = 3**3
   bar()

def bar():
   print("foo")

print(tsensor.__version__)
with tsensor.explain():
   a = np.ones(3)
   # foo()