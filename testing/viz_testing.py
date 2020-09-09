import sys
import torch
import graphviz
import tempfile

from tsensor.viz import pyviz_graphviz

W = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.tensor([1, 2])
a = 3

frame = sys._getframe()
html1 = pyviz_graphviz("b = W@b + h.dot(h) + torch.abs(torch.tensor(34))", frame)
html2 = pyviz_graphviz("x+4", frame)
print(html2)

g1 = graphviz.Source(html1)
g2 = graphviz.Source(html2)

g1.render(filename="g1", directory="/tmp", format="svg", quiet=True)
# g2.render(filename="g2", directory="/tmp")
# g1.view(tempfile.mktemp('.dot'))
g2.view(quiet=True)
