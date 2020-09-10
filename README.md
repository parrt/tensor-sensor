# Tensor Sensor

The goal of this library is to generate more helpful exception
messages for numpy/pytorch matrix algebra expressions.  Because the
matrix algebra in pytorch and numpy are all done in C/C++, they do not
have access to the Python execution environment so they are literally
unable to give information about which Python variables caused the
problem.  Only by catching the exception and then analyzing the Python
code can we get this kind of an error message.

The Python `with` statement allows me to trap exceptions that occur
and then I literally parse the Python code of the offending line, build an
expression tree, and then incrementally evaluate the operands
bottom-up until I run into an exception. That tells me which of the
subexpressions caused the problem and then I can pull it apart and
ask if any of those operands are matrices.

Imagine you have a complicated little matrix expression like:

```
W @ torch.dot(b,b)+ torch.eye(2,2)@x + z
```

And you get this unhelpful error message from pytorch:

```
RuntimeError: 1D tensors expected, got 2D, 2D tensors at [...]/THTensorEvenMoreMath.cpp:83
```

There are two problems: it does not tell you which of the sub
expressions threw the exception and it does not tell you what the
shape of relevant operands are.  This library that lets you
do this:

```
import tsensor
with tsensor.clarify():
    W @ torch.dot(b,b)+ torch.eye(2,2)@x + z
```

which then augments the exception message with the following clarification:

```
Cause: torch.dot(b,b) tensor arg b w/shape [2, 1], arg b w/shape [2, 1]
```

Here’s another default error message that is almost helpful for expression `W @ z`:

```
RuntimeError: size mismatch, get 2, 2x2,3
```

But tensor-sensor gives:

```
Cause: @ on tensor operand W w/shape [2, 2] and operand z w/shape [3]
```

Non-tensor args/values are ignored.

```
with tsensor.clarify():
    torch.dot(b, 3)
```

gives:

```
TypeError: dot(): argument 'tensor' (position 2) must be Tensor, not int
Cause: torch.dot(b,3) tensor arg b w/shape [2, 1]
```

If there are no tensor args, it just shows the cause:

```
with tsensor.clarify():
    z.reshape(1,2,2)
```

gives:

```
RuntimeError: shape '[1, 2, 2]' is invalid for input of size 3
Cause: z.reshape(1,2,2)
```

## Visualizations

For more, see [examples.ipynb](testing/examples.ipynb).

```python
import tsensor
import graphviz
import torch
import sys

W = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.tensor([1,2])

with tsensor.explain():
    a = torch.relu(x)
    b = W @ b + h.dot(h)
```

Displays this in a notebook:

<img src="images/sample-1.svg">

<img src="images/sample-2.svg">


## Install

```
pip install tensor-sensor
```

which gives you module `tsensor`.


## Deploy (parrt's use)

```bash
$ python setup.py sdist upload 
```

Or download and install locally

```bash
$ cd ~/github/tensor-sensor
$ pip install .
```