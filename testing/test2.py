import numpy as np
import tsensor
import torch

W = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([9, 10], dtype=int).reshape(2, 1)
x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.tensor([1,2])

with tsensor.clarify(legend=True):
    W @ torch.dot(b, b) + torch.eye(2, 2) @ x