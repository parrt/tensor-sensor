import numpy as np
import tsensor
import torch

W = np.array([[1, 2], [3, 4]])
b = np.array([9, 10]).reshape(2, 1)
x = np.array([4, 5]).reshape(2, 1)
h = np.array([1, 2])
with tsensor.explain(savefig="/Users/parrt/Desktop/foo.pdf"):
    W @ np.dot(b,b) + np.eye(2,2)@x


W = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5], dtype=torch.int32).reshape(2, 1)
h = torch.tensor([1,2])

a = torch.rand(size=(2, 20), dtype=torch.float64)
b = torch.rand(size=(2, 20), dtype=torch.float32)
c = torch.rand(size=(2,20,200), dtype=torch.complex64)
d = torch.rand(size=(2,20,200,5), dtype=torch.float16)


with tsensor.explain(savefig="/Users/parrt/Desktop/t2.pdf"):
    a + b + x + c[:,:,0] + d[:,:,0,0]

with tsensor.explain(savefig="/Users/parrt/Desktop/t3.pdf"):
    c

with tsensor.explain(savefig="/Users/parrt/Desktop/t4.pdf"):
    d

# with tsensor.explain(legend=True, savefig="/Users/parrt/Desktop/t.pdf") as e:
#     W @ torch.dot(b, b) + torch.eye(2, 2) @ x
