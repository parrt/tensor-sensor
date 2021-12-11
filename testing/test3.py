import matplotlib.pyplot as plt
import numpy as np
import tsensor
import torch
import sys

W = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5], dtype=torch.int32).reshape(2, 1)
h = torch.tensor([1,2])

# fig, ax = plt.subplots(1,1)
# # view = tsensor.pyviz("b + x", ax=ax, legend=True)
# # view.savefig("/Users/parrt/Desktop/foo.pdf")
# plt.show()

W = torch.rand(size=(2000,2000), dtype=torch.float64)
b = torch.rand(size=(2000,1), dtype=torch.float64)
h = torch.zeros(size=(1_000_000,), dtype=int)
x = torch.rand(size=(2000,1))
z = torch.rand(size=(2000,1), dtype=torch.complex64)
g = tsensor.astviz("b = W@b + (h+3).dot(h) + z",
                   sys._getframe()) # eval, highlight vectors
g.view()

# with tsensor.explain():
#     b + x

