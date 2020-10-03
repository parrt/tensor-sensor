import torch
import tsensor

n = 200                          # number of instances
d = 764                          # number of instance features
n_neurons = 100                  # how many neurons in this layer?
batch_size = 10                  # how many records per batch?
n_batches = n // batch_size

W = torch.rand(n_neurons,d)
b = torch.rand(n_neurons,1)
X = torch.rand(n_batches,batch_size,d)

with tsensor.explain():
    for i in range(n_batches):
        batch = X[i,:,:]
        Y = W @ batch.T + b 