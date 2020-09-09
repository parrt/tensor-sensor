import sys
import traceback
import trace
import inspect
import dis
from inspect import currentframe, getframeinfo, stack
import torch
import numpy as np

import tsensor


from collections import namedtuple
Foo = namedtuple("Foo", ["c", "d"])

class A:
    def __init__(self):
        self.b = Foo(33,"hi")
    def f(self):
        return 99

with tsensor.analysis():
    a = A()
    a.b
    a.f()
    a.b.c
    a.b.c.d()
    W = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([9, 10]).reshape(2, 1)
    z = W@b
