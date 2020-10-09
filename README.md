# Tensor Sensor

<img src="https://explained.ai/tensor-sensor/images/teaser.png" width="50%" align="right">One of the biggest challenges when writing code to implement deep learning networks, particularly for us newbies, is getting all of the tensor (matrix and vector) dimensions to line up properly. It's really easy to lose track of tensor dimensionality in complicated expressions involving multiple tensors and tensor operations.  Even when just feeding data into predefined [Tensorflow](https://www.tensorflow.org/) network layers, we still need to get the dimensions right. When you ask for improper computations, you're going to run into some less than helpful exception messages.  To help myself and other programmers debug tensor code, I built this library.  TensorSensor clarifies exceptions by augmenting messages and visualizing Python code to indicate the shape of tensor variables (see figure to the right for a teaser). It works with [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and [Numpy](https://numpy.org/), as well as higher-level libraries like [Keras](https://keras.io/) and [fastai](https://www.fast.ai/).

Please read the complete description in article [Clarifying exceptions and visualizing tensor operations in deep learning code](https://explained.ai/tensor-sensor/index.html).

*TensorSensor is currently at 0.1b1 so I'm happy to receive issues created at this repo or direct email*.

## Visualizations

For more, see [examples.ipynb](testing/examples.ipynb).

```python
import torch
W = torch.rand(d,n_neurons)
b = torch.rand(n_neurons,1)
X = torch.rand(n,d)
with tsensor.clarify():
    Y = W @ X.T + b
```

Displays this in a jupyter notebook or separate window:

<img src="https://explained.ai/tensor-sensor/images/mm.svg">

Instead of the following default exception message:

```
RuntimeError: size mismatch, m1: [764 x 100], m2: [764 x 200] at /tmp/pip-req-build-as628lz5/aten/src/TH/generic/THTensorMath.cpp:41
```

TensorSensor augments the message with more information about which operator caused the problem and includes the shape of the operands:

```
Cause: @ on tensor operand W w/shape [764, 100] and operand X.T w/shape [764, 200]
```


## Install

```
pip install tensor-sensor # This will only install the library for you
pip install tensor-sensor[torch] # install pytorch related dependency
pip install tensor-sensor[tensorflow] # install tensorflow related dependency
pip install tensor-sensor[all] # install both tensorflow and pytorch
```

which gives you module `tsensor`. I developed and tested with the following versions

```
$ pip list | grep -i flow
tensorflow                         2.3.0
tensorflow-estimator               2.3.0
$ pip list | grep -i numpy
numpy                              1.18.5
numpydoc                           1.1.0
$ pip list | grep -i torch
torch                              1.6.0
```


## Limitations

I rely on parsing lines that are assignments or expressions only so the clarify and explain routines do not handle methods expressed like:

```
def bar(): b + x * 3
```

Instead, use

```
def bar():
	b + x * 3
```

watch out for side effects!  I don't do assignments, but any functions you call with side effects will be done while I reevaluate statements.

Can't handle `\` continuations.

Also note: I've built my own parser to handle just the assignments / expressions tsensor can handle.

## Deploy (parrt's use)

```bash
$ python setup.py sdist upload 
```

Or download and install locally

```bash
$ cd ~/github/tensor-sensor
$ pip install .
```

### TODO

* can i call pyviz in debugger?
* try on real examples
* `dict(W=[3,0,1,2], b=[1,0])` that would indicate (300, 30, 60, 3) would best be displayed as (30,60,3, 300) and b would be  first dimension last and last dimension first
