import sys
import torch
import graphviz
import tempfile
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import tsensor
# from tsensor.viz import pyviz, astviz

class GRU:
    def __init__(self):
        self.W = torch.rand(size=(2000,2000))
        self.b = torch.rand(size=(20,1))
        # self.x = torch.tensor([4, 5]).reshape(2, 1)
        self.h = torch.rand(size=(1_000_000,))
        self.a = 3
        print(self.W.shape)

    def get(self):
        return torch.tensor([[1, 2], [3, 4]])

# W = torch.tensor([[1, 2], [3, 4]])
b = torch.rand(size=(2000,1))
# x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.rand(size=(1_000_000,))
a = 3

code = "b = g.W@b+torch.zeros(2000,1)+( h +3).dot(h)"
# code = "(h+3).dot(h)"
g = GRU()
g = tsensor.pyviz(code,
                  fontsize=24,
                  fontname='Consolas')
# g.view()
# plt.tight_layout()
plt.savefig("/tmp/t.pdf", dpi=200, bbox_inches='tight', pad_inches=0)

# with tsensor.explain() as e:
#     a = torch.relu(x)
#     b = W @ b + h.dot(h)


# g = GRU()
#
# g1 = tsensor.astviz("b = g.W@b + torch.eye(3,3)")
# g1.view()
# g1 = tsensor.pyviz("b = g.W@b")
# g1.view()
# g2 = tsensor.astviz("b = g.W@b + g.h.dot(g.h) + torch.abs(torch.tensor(34))")
#
# #g1.render(filename="g1", directory="/tmp", format="svg", quiet=True)
# # g2.render(filename="g2", directory="/tmp")
# # g1.view(tempfile.mktemp('.dot'))
# g2.view(quiet=True)
# # g2.view(quiet=True)
