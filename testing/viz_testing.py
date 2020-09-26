import sys
import torch
import numpy as np
import tensorflow as tf
import graphviz
import tempfile
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# print('\n'.join(str(f) for f in fm.fontManager.ttflist))
import tsensor
# from tsensor.viz import pyviz, astviz

def foo():
    # W = torch.rand(size=(2000, 2000))
    W = torch.rand(size=(2000, 2000, 10, 8))
    b = torch.rand(size=(2000, 1))
    h = torch.rand(size=(1_000_000,))
    x = torch.rand(size=(2000, 1))
    # g = tsensor.astviz("b = W@b + (h+3).dot(h) + torch.abs(torch.tensor(34))",
    #                    sys._getframe())
    frame = sys._getframe()
    frame = None
    g = tsensor.astviz("b = W[:,:,0,0]@b + (h+3).dot(h) + torch.abs(torch.tensor(34))",
                       frame)
    g.view()

#foo()

class GRU:
    def __init__(self):
        self.W = torch.rand(size=(2,20,2000,10))
        self.b = torch.rand(size=(20,1))
        # self.x = torch.tensor([4, 5]).reshape(2, 1)
        self.h = torch.rand(size=(1_000_000,))
        self.a = 3
        print(self.W.shape)
        print(self.W[:, :, 1].shape)

    def get(self):
        return torch.tensor([[1, 2], [3, 4]])

# W = torch.tensor([[1, 2], [3, 4]])
b = torch.rand(size=(2000,1))
h = torch.rand(size=(1_000_000,2))
x = torch.rand(size=(1_000_000,2))
a = 3

# foo = torch.rand(size=(2000,))
# torch.relu(foo)

g = GRU()

# with tsensor.clarify():
#     tf.constant([1,2]) @ tf.constant([1,3])


code = "b = g.W[0,:,:,1]@b+torch.zeros(200,1)+(h+3).dot(h)"
code = "torch.relu(foo)"
code = "np.dot(b,b)"
code = "b.T"
g = tsensor.pyviz(code, fontname='Courier New', fontsize=16, dimfontsize=9,
                  char_sep_scale=1.8, hush_errors=False)
plt.tight_layout()
plt.savefig("/tmp/t.svg", dpi=200, bbox_inches='tight', pad_inches=0)

# W = torch.tensor([[1, 2], [3, 4]])
# x = torch.tensor([4, 5]).reshape(2, 1)
# with tsensor.explain():
#     b = torch.rand(size=(2000,))
#     torch.relu(b)


# g = GRU()
#
# g1 = tsensor.astviz("b = g.W@b + torch.eye(3,3)")
# g1.view()
# g1 = tsensor.pyviz("b = g.W@b")
# g1.view()
# g2 = tsensor.astviz("b = g.W@b + g.h.dot(g.h) + torch.abs(torch.tensor(34))")
