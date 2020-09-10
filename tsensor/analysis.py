import sys
import traceback
import torch
import inspect
import graphviz
from IPython.display import display, SVG

import tsensor

class clarify:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            if not self.is_interesting_exception(exc_value):
                return
            # print("exception:", exc_value, exc_traceback)
            # traceback.print_tb(exc_traceback, limit=5, file=sys.stdout)
            exc_frame = self.deepest_frame(exc_traceback)
            module, name, filename, line, code = self.info(exc_frame)
            # print('info', module, name, filename, line, code)
            if code is not None:
                # could be internal like "__array_function__ internals" in numpy
                self.process_exception(code, exc_frame, exc_value)

    def process_exception(self, code, exc_frame, exc_value):
        augment = ""
        try:
            p = tsensor.parsing.PyExprParser(code)
            t = p.parse()
            try:
                t.eval(exc_frame)
            except tsensor.ast.IncrEvalTrap as exc:
                subexpr = exc.offending_expr
                # print("trap evaluating:\n", repr(subexpr), "\nin", repr(t))
                explanation = subexpr.explain()
                if explanation is not None:
                    augment = explanation
        except BaseException as e:
            print(f"exception while eval({code})", e)
            traceback.print_tb(e.__traceback__, limit=5, file=sys.stdout)
        # Reuse exception but overwrite the message
        exc_value.args = [exc_value.args[0] + "\n" + augment]

    def is_interesting_exception(self, e):
        sentinels = {'matmul', 'THTensorMath', 'tensor', 'tensors', 'dimension',
                     'not aligned', 'size mismatch', 'shape', 'shapes'}
        msg = e.args[0]
        return sum([s in msg for s in sentinels])>0

    def deepest_frame(self, exc_traceback):
        tb = exc_traceback
        # don't trace into internals of numpy etc... with filenames like '<__array_function__ internals>'
        while tb.tb_next != None and not tb.tb_next.tb_frame.f_code.co_filename.startswith('<'):
            tb = tb.tb_next
        return tb.tb_frame

    def info(self, frame):
        if hasattr(frame, '__name__'):
            module = frame.f_globals['__name__']
        else:
            module = None
        info = inspect.getframeinfo(frame)
        if info.code_context is not None:
            code = info.code_context[0].strip()
        else:
            code = None
        filename, line = info.filename, info.lineno
        name = info.function
        return module, name, filename, line, code

class TensorTracer:
    def __init__(self, savefig:str=None, format="svg", modules=['__main__'], filenames=[]):
        self.savefig = savefig
        self.format = format
        self.modules = modules
        self.filenames = filenames
        self.exceptions = set()
        self.linecount = 0

    def listener(self, frame, event, arg):
        module = frame.f_globals['__name__']
        if module not in self.modules:
            return

        info = inspect.getframeinfo(frame)
        filename, line = info.filename, info.lineno
        name = info.function
        if len(self.filenames)>0 and filename not in self.filenames:
            return

        if event=='call':
            self.call_listener(module, name, filename, line)
            return self.listener

        # TODO: ignore c_call etc...

        if event=='line':
            self.line_listener(module, name, filename, line, info, frame)

        return None

    def call_listener(self, module, name, filename, line):
        # print(f"A call encountered in {module}.{name}() at {filename}:{line}")
        pass

    def line_listener(self, module, name, filename, line, info, frame):
        code = info.code_context[0].strip()
        if code.startswith("sys.settrace(None)"):
            return
        self.linecount += 1
        p = tsensor.parsing.PyExprParser(code)
        t = p.parse()
        if t is not None:
            # print(f"A line encountered in {module}.{name}() at {filename}:{line}")
            # print("\t", code)
            # print("\t", repr(t))
            g = tsensor.viz.pyviz(code, frame)
            dotfilename = f"{self.savefig}-{self.linecount}.dot"
            svgfilename = f"{self.savefig}-{self.linecount}.svg"
            if self.savefig is not None:
                g.save(dotfilename)
                # g.render(format="svg", quiet=True, view=False)
                cmd = ["dot", f"-Tsvg", "-o", svgfilename, dotfilename]
                # print(' '.join(cmd))
                graphviz.backend.run(cmd, capture_output=True, check=True, quiet=True)
            else:
                display(SVG(g.pipe(format="svg", quiet=True)))
            # g.render(quiet=True)
            # if self.format=='svg':
            #     tmp = tempfile.gettempdir()
            #     svgfilename = os.path.join(tmp, f"DTreeViz_{os.getpid()}.svg")
            #     display(SVG(g))
            # else:
            #     display(g)


class explain:
    def __init__(self, savefig=None):
        self.savefig = savefig

    def __enter__(self, format="svg"):
        # print("ON trace")
        self.tracer = TensorTracer(self.savefig,format=format)
        sys.settrace(self.tracer.listener)
        frame = sys._getframe()
        prev = frame.f_back # get block wrapped in "with"
        prev.f_trace = self.tracer.listener
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.settrace(None)
        # print("OFF trace")


def eval(statement:str, frame=None) -> (tsensor.ast.ParseTreeNode, object):
    """
    Parse statement and return ast. Evaluate ast in context of
    frame if available, which sets the value field of all ast nodes.
    Overall result is in root.value.
    """
    p = tsensor.parsing.PyExprParser(statement)
    root = p.parse()
    if frame is not None:
        root.eval(frame)
    return root, root.value


def _shape(v):
    if hasattr(v, "shape"):
        if isinstance(v.shape, torch.Size):
            if len(v.shape)==0:
                return None
            return list(v.shape)
        return v.shape
    return None
