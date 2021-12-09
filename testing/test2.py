import numpy as np
import tsensor
import torch

W = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5], dtype=torch.int32).reshape(2, 1)
h = torch.tensor([1,2])

with tsensor.explain(legend=True, savefig="/Users/parrt/Desktop/t2.pdf"):
    torch.rand(size=(2,20,2000,10))

# with tsensor.explain(legend=True, savefig="/Users/parrt/Desktop/t.pdf") as e:
#     W @ torch.dot(b, b) + torch.eye(2, 2) @ x
