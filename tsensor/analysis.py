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
import hashlib

import matplotlib.pyplot as plt

import tsensor

class clarify:
    def __init__(self,
                 fontname='Consolas', fontsize=13,
                 dimfontname='Arial', dimfontsize=9, matrixcolor="#cfe2d4",
                 vectorcolor="#fefecd", char_sep_scale=1.8, fontcolor='#444443',
                 underline_color='#C2C2C2', ignored_color='#B4B4B4', error_op_color='#A40227',
                 show:(None,'viz')='viz',
                 hush_errors=True):
        """
        Augment tensor-related exceptions generated from numpy, pytorch, and tensorflow.
        Also display a visual representation of the offending Python line that
        shows the shape of tensors referenced by the code. All you have to do is wrap
        the outermost level of your code and clarify() will activate upon exception.

        Visualizations pop up in a separate window unless running from a notebook,
        in which case the visualization appears as part of the cell execution output.

        There is no runtime overhead associated with clarify() unless an exception occurs.

        The offending code is executed a second time, to identify which sub expressions
        are to blame. This implies that code with side effects could conceivably cause
        a problem, but since an exception has been generated, results are suspicious
        anyway.

        Example:

        import numpy as np
        import tsensor

        b = np.array([9, 10]).reshape(2, 1)
        with tsensor.clarify():
            np.dot(b,b) # tensor code or call to a function with tensor code

        See examples.ipynb for more examples.

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
        self.fontcolor, self.underline_color, self.ignored_color, \
        self.error_op_color, self.hush_errors = \
            fontname, fontsize, dimfontname, dimfontsize, \
            matrixcolor, vectorcolor, char_sep_scale, \
            fontcolor, underline_color, ignored_color, error_op_color, hush_errors

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
                self.view = tsensor.viz.pyviz(code, exc_frame,
                                              self.fontname, self.fontsize, self.dimfontname,
                                              self.dimfontsize, self.matrixcolor, self.vectorcolor,
                                              self.char_sep_scale, self.fontcolor,
                                              self.underline_color, self.ignored_color,
                                              self.error_op_color,
                                              hush_errors=self.hush_errors)
                if self.show=='viz':
                    self.view.show()
                augment_exception(exc_value, self.view.offending_expr)


class explain:
    def __init__(self,
                 fontname='Consolas', fontsize=13,
                 dimfontname='Arial', dimfontsize=9, matrixcolor="#cfe2d4",
                 vectorcolor="#fefecd", char_sep_scale=1.8, fontcolor='#444443',
                 underline_color='#C2C2C2', ignored_color='#B4B4B4', error_op_color='#A40227',
                 savefig=None,
                 hush_errors=True):
        """
        As the Python virtual machine executes lines of code, generate a
        visualization for tensor-related expressions using from numpy, pytorch,
        and tensorflow. The shape of tensors referenced by the code are displayed.

        Visualizations pop up in a separate window unless running from a notebook,
        in which case the visualization appears as part of the cell execution output.

        There is heavy runtime overhead associated with explain() as every line
        is executed twice: once by explain() and then another time by the interpreter
        as part of normal execution.

        Expressions with side effects can easily generate incorrect results. Due to
        this and the overhead, you should limit the use of this to code you're trying
        to debug.  Assignments are not evaluated by explain so code `x = ...` causes
        an assignment to x just once, during normal execution. This explainer
        knows the value of x and will display it but does not assign to it.

        Upon exception, execution will stop as usual but, like clarify(), explain()
        will augment the exception to indicate the offending sub expression. Further,
        the visualization will deemphasize code not associated with the offending
        sub expression. The sizes of relevant tensor values are still visualized.

        Example:

        import numpy as np
        import tsensor

        b = np.array([9, 10]).reshape(2, 1)
        with tsensor.explain():
            b + b # tensor code or call to a function with tensor code

        See examples.ipynb for more examples.

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
        self.fontcolor, self.underline_color, self.ignored_color, \
        self.error_op_color, self.hush_errors = \
            fontname, fontsize, dimfontname, dimfontsize, \
            matrixcolor, vectorcolor, char_sep_scale, \
            fontcolor, underline_color, ignored_color, error_op_color, hush_errors

    def __enter__(self):
        # print("ON trace")
        self.tracer = ExplainTensorTracer(self)
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
    def __init__(self, explainer):
        self.explainer = explainer
        self.exceptions = set()
        self.linecount = 0
        self.views = []
        # set of hashes for statements already visualized;
        # generate each combination of statement and shapes once
        self.done = set()

    def listener(self, frame, event, arg):
        module = frame.f_globals['__name__']
        info = inspect.getframeinfo(frame)
        filename, line = info.filename, info.lineno
        name = info.function

        if event=='line':
            self.line_listener(module, name, filename, line, info, frame)

        # By returning none, we prevent explain()'ing from descending into
        # invoked functions. In principle, we could allow a certain amount
        # of tracing but I'm not sure that would be super useful.
        return None

    def line_listener(self, module, name, filename, line, info, frame):
        code = info.code_context[0].strip()
        if code.startswith("sys.settrace(None)"):
            return

        # Don't generate a statement visualization more than once
        h = self.hash(code)
        if h in self.done:
            return
        self.done.add(h)

        p = tsensor.parsing.PyExprParser(code)
        t = p.parse()
        if t is not None:
            # print(f"A line encountered in {module}.{name}() at {filename}:{line}")
            # print("\t", code)
            # print("\t", repr(t))
            self.linecount += 1
            self.viz_statement(code, frame)

    def viz_statement(self, code, frame):
        view = tsensor.viz.pyviz(code, frame,
                                 self.explainer.fontname, self.explainer.fontsize,
                                 self.explainer.dimfontname,
                                 self.explainer.dimfontsize, self.explainer.matrixcolor,
                                 self.explainer.vectorcolor,
                                 self.explainer.char_sep_scale, self.explainer.fontcolor,
                                 self.explainer.underline_color, self.explainer.ignored_color,
                                 self.explainer.error_op_color,
                                 hush_errors=self.explainer.hush_errors)
        self.views.append(view)
        if self.explainer.savefig is not None:
            svgfilename = f"{self.explainer.savefig}-{self.linecount}.svg"
            view.savefig(svgfilename)
            view.filename = svgfilename
            plt.close()
        else:
            view.show()
        return view

    def hash(self, statement):
        """
        We want to avoid generating a visualization more than once.
        For now, assume that the code for a statement is the unique identifier.
        """
        return hashlib.md5(statement.encode('utf-8')).hexdigest()



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
    we cannot start at the leaves and walk upwards. Instead, pass a Boolean
    back to indicate whether this note or one of the descendents
    evaluates to a non-scalar.  Nodes in the tree that have matrix values and
    no matrix below are the ones to visualize.
    """
    nodes = []
    _smallest_matrix_subexpr(t, nodes)
    return nodes

def _smallest_matrix_subexpr(t, nodes) -> bool:
    if t is None: return False  # prevent buggy code from causing us to fail
    if isinstance(t, tsensor.ast.Member) and \
       isinstance(t.obj, tsensor.ast.Atom) and \
       isinstance(t.member, tsensor.ast.Atom) and \
       str(t.member)=='T':
        nodes.append(t)
        return True
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
