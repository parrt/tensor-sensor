import sys
import traceback
import trace
import inspect
import dis
from inspect import currentframe, getframeinfo, stack

import numpy as np

# class Watcher(object):
#     def __init__(self):
#         pass
#     def trace_command(self, frame, event, arg):
#         print("trace")
#
# watcher = Watcher()

class Tracer:
    def __init__(self, modules=['__main__']):
        self.modules = modules

    def mytrace(self, frame, event, arg):
        info = inspect.getframeinfo(frame)
        module = frame.f_globals['__name__']
        if module not in self.modules:
            return self.mytrace
        filename, line = info.filename, info.lineno
        name = info.function

        # print(f"A {event} encountered in {module}.{name}() at {filename}:{line}")

        if event=='line':
            code = info.code_context[0].strip()
            if not code.startswith("def "):
                print(f"A {event} encountered in {module}.{name}() at {filename}:{line}")
                print("\t", code)
                c = dis.Bytecode(code)
                print(c.dis())
        return self.mytrace

class dbg:
    def __enter__(self):
        print("ON trace")
        tr = Tracer()
        sys.settrace(tr.mytrace)
        frame = sys._getframe()
        prev = frame.f_back
        # if frame.f_back is None:
        # prev = stack()[2]
        prev.f_trace = tr.mytrace

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.settrace(None)
        print("OFF trace")


def f():
    with dbg():
        W = np.array([[1, 2], [3, 4]])
        b = np.array([9, 10]).reshape(2, 1)
        x = np.array([4, 5]).reshape(2, 1)
        b = np.abs( W @ b + x )

# f()

with dbg():
    W = np.array([[1, 2], [3, 4]])
    b = np.array([9, 10]).reshape(2, 1)
    x = np.array([4, 5]).reshape(2, 1)
    b = np.abs( W.dot(b) + -x + np.pi + q() * self.t )

# tracer = trace.Trace(
#     ignoredirs=[sys.prefix, sys.exec_prefix],
#     trace=0,
#     count=1)
#
# tracer.run('f()')
#
# # make a report, placing output in the current directory
# r = tracer.results()
# print(r)
# r.write_results(show_missing=True, coverdir=".")