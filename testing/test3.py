import sys
import traceback
import trace
import inspect
import dis
from inspect import currentframe, getframeinfo, stack
import torch
import numpy as np

import tsensor

class Tracer:
    def __init__(self, modules=['__main__']):
        self.modules = modules
        self.exceptions = set()

    def listener(self, frame, event, arg):
        module = frame.f_globals['__name__']
        if module not in self.modules:
            return self.listener

        info = inspect.getframeinfo(frame)
        filename, line = info.filename, info.lineno
        name = info.function

        if event=='call':
            self.call_listener(module, name, filename, line)
            return self.listener

        # TODO: ignore c_call etc...

        if event=='line':
            self.line_listener(module, name, filename, line, info)

        return None

    def call_listener(self, module, name, filename, line):
        # print(f"A call encountered in {module}.{name}() at {filename}:{line}")
        pass

    def line_listener(self, module, name, filename, line, info):
        code = info.code_context[0].strip()
        if not code.startswith("def "):
            print(f"A line encountered in {module}.{name}() at {filename}:{line}")
            print("\t", code)
            p = tsensor.PyExprParser(code)
            t = p.parse()
            print("\t", repr(t))

            # c = dis.Bytecode(code)
            # print(c.dis())
            pass


class explainer:
    def __enter__(self):
        print("ON trace")
        tr = Tracer()
        sys.settrace(tr.listener)
        frame = sys._getframe()
        prev = frame.f_back
        # if frame.f_back is None:
        # prev = stack()[2]
        prev.f_trace = tr.listener

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.settrace(None)
        print("OFF trace")

from collections import namedtuple
Foo = namedtuple("Foo", ["c", "d"])

class A:
    def __init__(self):
        self.b = Foo(33,"hi")
    def f(self):
        return 99

with explainer():
    a = A()
    a.b
    a.f()
    a.b.c
    a.b.c.d()
    W = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([9, 10]).reshape(2, 1)
    z = W@b
