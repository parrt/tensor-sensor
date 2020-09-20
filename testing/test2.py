import tsensor
import numpy as np
import torch
import matplotlib.pyplot as plt

W = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.tensor([1,2])


# tsensor.pyviz("a = torch.relu(x)")
# plt.show()
# #

with tsensor.clarify():
    W @ np.dot(b, b) + np.eye(2, 2) @ x
#    b = W @ b + x * 3 + h.dot(h)