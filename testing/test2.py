import torch
import tsensor
nhidden = 256
n = 200         # number of instances
d = 764         # number of instance features
n_neurons = 100 # how many neurons in this layer?

Whh_ = torch.eye(nhidden, nhidden)
Uxh_ = torch.randn(nhidden, d)
bh_  = torch.zeros(nhidden, 1)
h = torch.randn(nhidden, 1)  # fake previous hidden state h
r = torch.randn(nhidden, 1)  # fake this computation
X = torch.rand(n,d)          # fake input

g = tsensor.astviz("h_ = torch.tanh(Whh_ @ (r*h) + Uxh_ @ X.T + bh_)")
g.savefig("/tmp/torch-gru-ast-shapes.svg")