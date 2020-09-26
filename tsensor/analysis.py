"""
MIT License

Copyright (c) 2020 Terence Parr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import sys
import traceback
import torch
import inspect

import matplotlib.pyplot as plt

import tsensor

class clarify:
    def __init__(self,
                 fontname='Consolas', fontsize=13,
                 dimfontname='Arial', dimfontsize=9, matrixcolor="#cfe2d4",
                 vectorcolor="#fefecd", char_sep_scale=1.8, fontcolor='#444443',
                 underline_color='#C2C2C2', ignored_color='#B4B4B4', error_op_color='#A40227',
                 show:(None,'viz')='viz'):
        """
        :param fontname: The name of the font used to display Python code
        :param fontsize: The font size used to display Python code; default is 13.
                         Also use this to increase the size of the generated figure;
                         larger font size means larger image.
        :param dimfontname:  The name of the font used to display the dimensions on the matrix and vector boxes
        :param dimfontsize: The  size of the font used to display the dimensions on the matrix and vector boxes
        :param matrixcolor:  The  color of matrix boxes
        :param vectorcolor: The color of vector boxes; only for tensors whose shape is (n,).
        :param char_sep_scale: It is notoriously difficult to discover how wide and tall
                               text is when plotted in matplotlib. In fact there's probably,
                               no hope to discover this information accurately in all cases.
                               Certainly, I gave up after spending huge effort. We have a
                               situation here where the font should be constant width, so
                               we can just use a simple scaler times the font size  to get
                               a reasonable approximation to the width and height of a
                               character box; the default of 1.8 seems to work reasonably
                               well for a wide range of fonts, but you might have to tweak it
                               when you change the font size.
        :param fontcolor:  The color of the Python code.
        :param underline_color:  The color of the lines that underscore tensor subexpressions; default is grey
        :param ignored_color: The de-highlighted color for deemphasizing code not involved in an erroneous sub expression
        :param error_op_color: The color to use for characters associated with the erroneous operator
        :param ax: If not none, this is the matplotlib drawing region in which to draw the visualization
        :param dpi: This library tries to generate SVG files, which are vector graphics not
                    2D arrays of pixels like PNG files. However, it needs to know how to
                    compute the exact figure size to remove padding around the visualization.
                    Matplotlib uses inches for its figure size and so we must convert
                    from pixels or data units to inches, which means we have to know what the
                    dots per inch, dpi, is for the image.
        :param hush_errors: Normally, error messages from true syntax errors but also
                            unhandled code caught by my parser are ignored. Turn this off
                            to see what the error messages are coming from my parser.
        :param show: Show visualization upon tensor error if show='viz'.
        """
        self.show = show
        self.fontname, self.fontsize, self.dimfontname, self.dimfontsize, \
        self.matrixcolor, self.vectorcolor, self.char_sep_scale,\
        self.fontcolor, self.underline_color, self.ignored_color, self.error_op_color = \
            fontname, fontsize, dimfontname, dimfontsize, \
            matrixcolor, vectorcolor, char_sep_scale, \
            fontcolor, underline_color, ignored_color, error_op_color

    def __enter__(self):
        self.frame = sys._getframe().f_back # where do we start tracking
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None and is_interesting_exception(exc_value):
            # print("exception:", exc_value, exc_traceback)
            # traceback.print_tb(exc_traceback, limit=5, file=sys.stdout)
            exc_frame = deepest_frame(exc_traceback)
            module, name, filename, line, code = info(exc_frame)
            # print('info', module, name, filename, line, code)
            if code is not None:
                view = tsensor.viz.pyviz(code, exc_frame,
                                         self.fontname, self.fontsize, self.dimfontname,
                                         self.dimfontsize, self.matrixcolor, self.vectorcolor,
                                         self.char_sep_scale, self.fontcolor,
                                         self.underline_color, self.ignored_color,
                                         self.error_op_color)
                if self.show=='viz':
                    view.show()
                augment_exception(exc_value, view.offending_expr)


class explain:
    def __init__(self,
                 fontname='Consolas', fontsize=13,
                 dimfontname='Arial', dimfontsize=9, matrixcolor="#cfe2d4",
                 vectorcolor="#fefecd", char_sep_scale=1.8, fontcolor='#444443',
                 underline_color='#C2C2C2', ignored_color='#B4B4B4', error_op_color='#A40227',
                 savefig=None):
        """
        :param fontname: The name of the font used to display Python code
        :param fontsize: The font size used to display Python code; default is 13.
                         Also use this to increase the size of the generated figure;
                         larger font size means larger image.
        :param dimfontname:  The name of the font used to display the dimensions on the matrix and vector boxes
        :param dimfontsize: The  size of the font used to display the dimensions on the matrix and vector boxes
        :param matrixcolor:  The  color of matrix boxes
        :param vectorcolor: The color of vector boxes; only for tensors whose shape is (n,).
        :param char_sep_scale: It is notoriously difficult to discover how wide and tall
                               text is when plotted in matplotlib. In fact there's probably,
                               no hope to discover this information accurately in all cases.
                               Certainly, I gave up after spending huge effort. We have a
                               situation here where the font should be constant width, so
                               we can just use a simple scaler times the font size  to get
                               a reasonable approximation to the width and height of a
                               character box; the default of 1.8 seems to work reasonably
                               well for a wide range of fonts, but you might have to tweak it
                               when you change the font size.
        :param fontcolor:  The color of the Python code.
        :param underline_color:  The color of the lines that underscore tensor subexpressions; default is grey
        :param ignored_color: The de-highlighted color for deemphasizing code not involved in an erroneous sub expression
        :param error_op_color: The color to use for characters associated with the erroneous operator
        :param ax: If not none, this is the matplotlib drawing region in which to draw the visualization
        :param dpi: This library tries to generate SVG files, which are vector graphics not
                    2D arrays of pixels like PNG files. However, it needs to know how to
                    compute the exact figure size to remove padding around the visualization.
                    Matplotlib uses inches for its figure size and so we must convert
                    from pixels or data units to inches, which means we have to know what the
                    dots per inch, dpi, is for the image.
        :param hush_errors: Normally, error messages from true syntax errors but also
                            unhandled code caught by my parser are ignored. Turn this off
                            to see what the error messages are coming from my parser.
        :param savefig: A string indicating where to save the visualization; don't save
                        a file if None.
        """
        self.savefig = savefig
        self.fontname, self.fontsize, self.dimfontname, self.dimfontsize, \
        self.matrixcolor, self.vectorcolor, self.char_sep_scale,\
        self.fontcolor, self.underline_color, self.ignored_color, self.error_op_color = \
            fontname, fontsize, dimfontname, dimfontsize, \
            matrixcolor, vectorcolor, char_sep_scale, \
            fontcolor, underline_color, ignored_color, error_op_color

    def __enter__(self, format="svg"):
        # print("ON trace")
        self.tracer = ExplainTensorTracer(self.savefig, format=format)
        sys.settrace(self.tracer.listener)
        frame = sys._getframe()
        prev = frame.f_back # get block wrapped in "with"
        prev.f_trace = self.tracer.listener
        return self.tracer

    def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.settrace(None)
        # At this point we have already tried to visualize the statement
        # If there was no error, the visualization will look normal
        # but a matrix operation error will show the erroneous operator highlighted.
        # That was artificial execution of the code. Now the VM has executed
        # the statement for real and has found the same exception. Make sure to
        # augment the message with causal information.
        if exc_type is not None and is_interesting_exception(exc_value):
            # print("exception:", exc_value, exc_traceback)
            # traceback.print_tb(exc_traceback, limit=5, file=sys.stdout)
            exc_frame = deepest_frame(exc_traceback)
            module, name, filename, line, code = info(exc_frame)
            # print('info', module, name, filename, line, code)
            if code is not None:
                # We've already displayed picture so just augment message
                root, tokens = tsensor.parsing.parse(code)
                if root is not None: # Could be syntax error in statement or code I can't handle
                    offending_expr = None
                    try:
                        root.eval(exc_frame)
                    except tsensor.ast.IncrEvalTrap as e:
                        offending_expr = e.offending_expr
                    augment_exception(exc_value, offending_expr)


class ExplainTensorTracer:
    def __init__(self, savefig:str=None, format="svg", modules=['__main__'], filenames=[]):
        self.savefig = savefig
        self.format = format
        self.modules = modules
        self.filenames = filenames
        self.exceptions = set()
        self.linecount = 0
        self.views = []

    def listener(self, frame, event, arg):
        module = frame.f_globals['__name__']
        if module not in self.modules:
            return

        info = inspect.getframeinfo(frame)
        filename, line = info.filename, info.lineno
        name = info.function
        if len(self.filenames)>0 and filename not in self.filenames:
            return

        if event=='line':
            self.line_listener(module, name, filename, line, info, frame)

        return None

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
            ExplainTensorTracer.viz_statement(self, code, frame)

    @staticmethod
    def viz_statement(tracer, code, frame):
        view = tsensor.viz.pyviz(code, frame)
        tracer.views.append(view)
        if tracer.savefig is not None:
            svgfilename = f"{tracer.savefig}-{tracer.linecount}.svg"
            view.savefig(svgfilename)
            view.filename = svgfilename
            plt.close()
        else:
            view.show()
        return view


def eval(statement:str, frame=None) -> (tsensor.ast.ParseTreeNode, object):
    """
    Parse statement and return an ast in the context of execution frame or, if None,
    the invoking function's frame. Set the value field of all ast nodes.
    Overall result is in root.value.
    :param statement: A string representing the line of Python code to visualize within an execution frame.
    :param frame: The execution frame in which to evaluate the statement. If None,
                  use the execution frame of the invoking function
    :return An abstract parse tree representing the statement; nodes are
            ParseTreeNode subclasses.
    """
    p = tsensor.parsing.PyExprParser(statement)
    root = p.parse()
    if frame is None: # use frame of caller
        frame = sys._getframe().f_back
    root.eval(frame)
    return root, root.value


def augment_exception(exc_value, subexpr):
    explanation = subexpr.clarify()
    augment = ""
    if explanation is not None:
        augment = explanation
    # Reuse exception but overwrite the message
    # print(f"Exc type is {type(exc_value)}, len(args)={len(exc_value.args)}, has '_message'=={hasattr(exc_value, '_message')}")
    # print(f"Msg {str(exc_value)}")
    if hasattr(exc_value, "_message"):
        exc_value._message = exc_value.message + "\n" + augment
    else:
        exc_value.args = [exc_value.args[0] + "\n" + augment]


def is_interesting_exception(e):
    # print(f"is_interesting_exception: type is {type(e)}")
    if e.__class__.__module__.startswith("tensorflow"):
        return True
    sentinels = {'matmul', 'THTensorMath', 'tensor', 'tensors', 'dimension',
                 'not aligned', 'size mismatch', 'shape', 'shapes', 'matrix'}
    if len(e.args)==0:
        msg = e.message
    else:
        msg = e.args[0]
    return sum([s in msg for s in sentinels])>0


def deepest_frame(exc_traceback):
    """
    Don't trace into internals of numpy/torch/tensorflow; we want to reset frame
    to where in the user's python code it asked the tensor lib to perform an
    invalid operation.

    To detect libraries, look for code whose filename has "site-packages/{package}"
    or "dist-packages/{package}".
    """
    tb = exc_traceback
    packages = ['numpy','torch','tensorflow']
    dirs = [os.path.join('site-packages',p) for p in packages]
    dirs += [os.path.join('dist-packages',p) for p in packages]
    dirs += ['<__array_function__'] # numpy seems to not have real filename
    prev = tb
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        reached_lib = [p in filename for p in dirs]
        if sum(reached_lib)>0:
            break
        prev = tb
        tb = tb.tb_next
    return prev.tb_frame


def info(frame):
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


def smallest_matrix_subexpr(t):
    """
    During visualization, we need to find the smallest expression
    that evaluates to a non-scalar. That corresponds to the deepest subtree
    that evaluates to a non-scalar. Because we do not have parent pointers,
    we cannot start at the leaves and walk upwards. Instead, set a Boolean
    in each node to indicate whether one of the descendents (but not itself)
    evaluates to a non-scalar.  Nodes in the tree that have matrix values and
    not matrix_below are the ones to visualize.

    This routine modifies the tree nodes to turn on matrix_below where appropriate.
    """
    nodes = []
    _smallest_matrix_subexpr(t, nodes)
    return nodes

def _smallest_matrix_subexpr(t, nodes) -> bool:
    if t is None: return False  # prevent buggy code from causing us to fail
    if len(t.kids)==0: # leaf node
        if istensor(t.value):
            nodes.append(t)
        return istensor(t.value)
    n_matrix_below = 0 # once this latches true, it's passed all the way up to the root
    for sub in t.kids:
        matrix_below = _smallest_matrix_subexpr(sub, nodes)
        n_matrix_below += matrix_below # how many descendents evaluated two non-scalar?
    # If current node is matrix and no descendents are, then this is smallest
    # sub expression that evaluates to a matrix; keep track
    if istensor(t.value) and n_matrix_below==0:
        nodes.append(t)
    # Report to caller that this node or some descendent is a matrix
    return istensor(t.value) or n_matrix_below > 0


def istensor(x):
    return _shape(x) is not None


def _shape(v):
    # do we have a shape and it answers len()? Should get stuff right.
    if hasattr(v, "shape") and hasattr(v.shape,"__len__"):
        if isinstance(v.shape, torch.Size):
            if len(v.shape)==0:
                return None
            return list(v.shape)
        return v.shape
    return None
