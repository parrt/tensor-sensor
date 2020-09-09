import tsensor
import numpy as np
import torch

def f():
    W = np.array([[1, 2], [3, 4]])
    b = np.array([9, 10]).reshape(2, 1)
    x = np.array([4, 5]).reshape(2, 1)
    z = np.array([1,2,3])
    # z + z + W @ z
    # W @ z
    # np.dot(b, b)
    W @ np.dot(b,b)+ np.eye(2,2)@x + z
    # W[33, 33] = 3
    b = np.abs( W @ b + x )

def g():
    W = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([9, 10]).reshape(2, 1)
    x = torch.tensor([4, 5]).reshape(2, 1)
    z = torch.tensor([1,2,3])
    # z + z + W @ z
    # W @ z
    torch.dot(b, 3)
    W @ torch.dot(b,b)+ torch.eye(2,2)@x + z
    # W[33, 33] = 3
    b = torch.abs( W @ b + x )

# tr = Tracer()
# sys.settrace(tr.listener)
# frame = sys._getframe()
# frame.f_trace = tr.listener

def foo():
    g()


with tsensor.clarify():
    g()

