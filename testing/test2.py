import tsensor
import numpy as np
import sys

W = np.array([[1, 2], [3, 4]])
b = np.array([9, 10]).reshape(2, 1)
x = np.array([4, 5]).reshape(2, 1)
h = np.array([1,2])

with tsensor.clarify():
    np.dot(b, b)# + np.eye(2, 2) @ x