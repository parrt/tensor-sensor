import sys
import torch
import graphviz
import tempfile

import tsensor
# from tsensor.viz import pyviz, astviz

class GRU:
    def __init__(self):
        self.W = torch.tensor([[1, 2], [3, 4], [5, 6]])
        self.b = torch.tensor([9, 10]).reshape(2, 1)
        self.x = torch.tensor([4, 5]).reshape(2, 1)
        self.h = torch.tensor([1, 2])
        self.a = 3


W = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.tensor([1, 2])
a = 3

with tsensor.explain(savefig="/tmp/sample") as e:
    a = torch.relu(x)
    b = W @ b + h.dot(h)


# g = GRU()
#
# frame = sys._getframe()
# g1 = tsensor.pyviz("b = g.W@b + g.h.dot(g.h) + torch.abs(torch.tensor(34))", frame)
# g2 = tsensor.astviz("b = g.W@b + g.h.dot(g.h) + torch.abs(torch.tensor(34))")
#
# #g1.render(filename="g1", directory="/tmp", format="svg", quiet=True)
# # g2.render(filename="g2", directory="/tmp")
# # g1.view(tempfile.mktemp('.dot'))
# g2.view(quiet=True)
# # g2.view(quiet=True)
