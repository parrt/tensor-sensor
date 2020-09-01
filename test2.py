import sys
import traceback
import trace
import inspect
import dis
from inspect import currentframe, getframeinfo, stack
import torch

import numpy as np

from matricks import dbg
import matricks

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

        if event=='exception':
            """
            From manual: "Note that as an exception is propagated down the chain
            of callers, an 'exception' event is generated at each level.
            """
            self.exception_listener(module, name, filename, line, frame, arg)
            return None

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
            # print(f"A line encountered in {module}.{name}() at {filename}:{line}")
            # print("\t", code)
            # c = dis.Bytecode(code)
            # print(c.dis())
            pass

    def exception_listener(self, module, name, filename, line, frame, arg):
        exc_type, exc_value, exc_traceback = arg
        if exc_value in self.exceptions: # already processed
            return
        print(f"\nException {id(arg)} encountered in {module}.{name}() at {filename}:{line}")
        self.exceptions.add(exc_value)
        print(f"EXC{exc_type, id(exc_value), exc_value}")
        # traceback.print_tb(exc_traceback, limit=2, file=sys.stdout)
        # print("Frame", frame, "locals", frame.f_locals)
        try:
            ugh = eval("W",frame.f_locals,frame.f_globals)
            print("W=",ugh)
            pass
        except Exception as ee:
            print("what?", ee)


# class dbg:
#     def __enter__(self):
#         print("ON trace")
#         tr = Tracer()
#         sys.settrace(tr.listener)
#         frame = sys._getframe()
#         prev = frame.f_back
#         # if frame.f_back is None:
#         # prev = stack()[2]
#         prev.f_trace = tr.listener
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         sys.settrace(None)
#         print("OFF trace")



def f():
    W = np.array([[1, 2], [3, 4]])
    W @ np.array([[1,2,3]])
    W[33, 33] = 3
    b = np.array([9, 10]).reshape(2, 1)
    x = np.array([4, 5]).reshape(2, 1)
    b = np.abs( W @ b + x )

def g():
    W = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([9, 10]).reshape(2, 1)
    W @ torch.tensor([[1,2,3]])
    W[33, 33] = 3
    x = torch.tensor([4, 5]).reshape(2, 1)
    b = np.abs( W @ b + x )

# tr = Tracer()
# sys.settrace(tr.listener)
# frame = sys._getframe()
# frame.f_trace = tr.listener

def foo():
    g()

with matricks.dbg():
    foo()
