# matricks

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
import matricks
with matricks.dbg():
    W @ torch.dot(b,b)+ torch.eye(2,2)@x + z
```

which then emits the following better error message:

```
Call torch.dot(b,b) has arg b w/shape torch.Size([2, 1]), arg b w/shape torch.Size([2, 1])
```

Here’s another default error message that is almost helpful for expression W @ z:

```
RuntimeError: size mismatch, get 2, 2x2,3
```

But this library gives:

```
Operation @ has operand W w/shape torch.Size([2, 2]) and operand z w/shape torch.Size([3])
```
