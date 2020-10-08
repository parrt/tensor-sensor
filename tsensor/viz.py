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
import sys
import os
from pathlib import Path
import tempfile
import graphviz
import graphviz.backend
import token
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from IPython.display import display, SVG
from IPython import get_ipython

import numpy as np
import tsensor
import tsensor.ast
import tsensor.analysis
import tsensor.parsing


class PyVizView:
    """
    An object that collects relevant information about viewing Python code
    with visual annotations.
    """
    def __init__(self, statement, fontname, fontsize, dimfontname, dimfontsize,
                 matrixcolor, vectorcolor, char_sep_scale, dpi):
        self.statement = statement
        self.fontsize = fontsize
        self.fontname = fontname
        self.dimfontsize = dimfontsize
        self.dimfontname = dimfontname
        self.matrixcolor = matrixcolor
        self.vectorcolor = vectorcolor
        self.char_sep_scale = char_sep_scale
        self.dpi = dpi
        self.wchar = self.char_sep_scale * self.fontsize
        self.hchar = self.char_sep_scale * self.fontsize
        self.dim_ypadding = 5
        self.dim_xpadding = 0
        self.linewidth = .7
        self.leftedge = 25
        self.bottomedge = 3
        self.filename = None
        self.matrix_size_scaler = 3.5      # How wide or tall as scaled fontsize is matrix?
        self.vector_size_scaler = 3.2 / 4  # How wide or tall as scaled fontsize is vector for skinny part?
        self.shift3D = 6
        self.cause = None # Did an exception occurred during evaluation?
        self.offending_expr = None
        self.fignumber = None

    def set_locations(self, maxh):
        """
        This function finishes setting up necessary parameters about text
        and graphics locations for the plot. We don't know how to set these
        values until we know what the max height of the drawing will be. We don't know
        what that height is until after we've parsed and so on, which requires that
        we collect and store information in this view object before computing maxh.
        That is why this is a separate function not part of the constructor.
        """
        line2text = self.hchar / 1.7
        box2line  = line2text*2.6
        self.texty = self.bottomedge + maxh + box2line + line2text
        self.liney = self.bottomedge + maxh + box2line
        self.box_topy  = self.bottomedge + maxh
        self.maxy = self.texty + 1.4 * self.fontsize

    def _repr_svg_(self):
        "Show an SVG rendition in a notebook"
        return self.svg()

    def svg(self):
        """
        Render as svg and return svg text. Save file and store name in field svgfilename.
        """
        if self.filename is None: # have we saved before? (i.e., is it cached?)
            self.savefig(tempfile.mktemp(suffix='.svg'))
        elif not self.filename.endswith(".svg"):
            return None
        with open(self.filename, encoding='UTF-8') as f:
            svg = f.read()
        return svg

    def savefig(self, filename):
        "Save viz in format according to file extension."
        if plt.fignum_exists(self.fignumber):
            # If the matplotlib figure is still active, save it
            self.filename = filename # Remember the file so we can pull it back
            plt.savefig(filename, dpi = self.dpi, bbox_inches = 'tight', pad_inches = 0)
        else: # we have already closed it so try to copy to new filename from the previous
            if filename!=self.filename:
                f,ext = os.path.splitext(filename)
                prev_f,prev_ext = os.path.splitext(self.filename)
                if ext != prev_ext:
                    print(f"File extension {ext} differs from previous {prev_ext}; uses previous.")
                    ext = prev_ext
                filename = f+ext # make sure that we don't copy raw bits and change the file extension to be inconsistent
                with open(self.filename, 'rb') as f:
                    img = f.read()
                with open(filename, 'wb') as f:
                    f.write(img)
                self.filename = filename  # overwrite the filename with new name

    def show(self):
        "Display an SVG in a notebook or pop up a window if not in notebook"
        if get_ipython() is None:
            svgfilename = tempfile.mktemp(suffix='.svg')
            self.savefig(svgfilename)
            self.filename = svgfilename
            plt.show()
        else:
            svg = self.svg()
            display(SVG(svg))
        plt.close()

    def boxsize(self, v):
        """
        How wide and tall should we draw the box representing a vector or matrix.
        """
        sh = tsensor.analysis._shape(v)
        if sh is None: return None
        if len(sh)==1: return self.vector_size(sh)
        return self.matrix_size(sh)

    def matrix_size(self, sh):
        """
        How wide and tall should we draw the box representing a matrix.
        """
        if len(sh)==1 and sh[0]==1:
            return self.vector_size(sh)
        elif len(sh) > 1 and sh[0] == 1 and sh[1] == 1:
            # A special case where we have a 1x1 matrix extending into the screen.
            # Make the 1x1 part a little bit wider than a vector so it's more readable
            return (2*self.vector_size_scaler * self.wchar, 2*self.vector_size_scaler * self.wchar)
        elif len(sh) > 1 and sh[1] == 1:
            return (
            self.vector_size_scaler * self.wchar, self.matrix_size_scaler * self.wchar)
        elif len(sh)>1 and sh[0]==1:
            return (self.matrix_size_scaler * self.wchar, self.vector_size_scaler * self.wchar)
        return (self.matrix_size_scaler * self.wchar, self.matrix_size_scaler * self.wchar)

    def vector_size(self, sh):
        return (self.matrix_size_scaler * self.wchar, self.vector_size_scaler * self.wchar)

    def draw(self, ax, sub):
        sh = tsensor.analysis._shape(sub.value)
        if len(sh)==1: self.draw_vector(ax, sub)
        else: self.draw_matrix(ax, sub)

    def draw_vector(self,ax,sub):
        a, b = sub.leftx, sub.rightx
        mid = (a + b) / 2
        sh = tsensor.analysis._shape(sub.value)
        w,h = self.vector_size(sh)
        rect1 = patches.Rectangle(xy=(mid - w/2, self.box_topy-h),
                                  width=w,
                                  height=h,
                                  linewidth=self.linewidth,
                                  facecolor=self.vectorcolor,
                                  edgecolor='grey',
                                  fill=True)
        ax.add_patch(rect1)
        ax.text(mid, self.box_topy + self.dim_ypadding, self.nabbrev(sh[0]),
                horizontalalignment='center',
                fontname=self.dimfontname, fontsize=self.dimfontsize)

    def draw_matrix(self,ax,sub):
        a, b = sub.leftx, sub.rightx
        mid = (a + b) / 2
        sh = tsensor.analysis._shape(sub.value)
        w,h = self.matrix_size(sh)
        box_left = mid - w / 2
        if len(sh)>2:
            back_rect = patches.Rectangle(xy=(box_left + self.shift3D, self.box_topy - h + self.shift3D),
                                          width=w,
                                          height=h,
                                          linewidth=self.linewidth,
                                          facecolor=self.matrixcolor,
                                          edgecolor='grey',
                                          fill=True)
            ax.add_patch(back_rect)
        rect = patches.Rectangle(xy=(box_left, self.box_topy - h),
                                  width=w,
                                  height=h,
                                  linewidth=self.linewidth,
                                  facecolor=self.matrixcolor,
                                  edgecolor='grey',
                                  fill=True)
        ax.add_patch(rect)
        ax.text(box_left, self.box_topy - h/2, self.nabbrev(sh[0]),
                verticalalignment='center', horizontalalignment='right',
                fontname=self.dimfontname, fontsize=self.dimfontsize, rotation=90)
        if len(sh)>1:
            textx = mid
            texty = self.box_topy + self.dim_ypadding
            if len(sh) > 2:
                texty += self.dim_ypadding
                textx += self.shift3D
            ax.text(textx, texty, self.nabbrev(sh[1]), horizontalalignment='center',
                    fontname=self.dimfontname, fontsize=self.dimfontsize)
        if len(sh)>2:
            ax.text(box_left+w, self.box_topy - h/2, self.nabbrev(sh[2]),
                    verticalalignment='center', horizontalalignment='center',
                    fontname=self.dimfontname, fontsize=self.dimfontsize,
                    rotation=45)
        if len(sh)>3:
            remaining = "$\cdots$x"+'x'.join([self.nabbrev(sh[i]) for i in range(3,len(sh))])
            ax.text(mid, self.box_topy - h - self.dim_ypadding, remaining,
                    verticalalignment='top', horizontalalignment='center',
                    fontname=self.dimfontname, fontsize=self.dimfontsize)

    @staticmethod
    def nabbrev(n) -> str:
        if n % 1_000_000 == 0:
            return str(n // 1_000_000)+'m'
        if n % 1_000 == 0:
            return str(n // 1000)+'k'
        return str(n)


def pyviz(statement: str, frame=None,
          fontname='Consolas', fontsize=13,
          dimfontname='Arial', dimfontsize=9, matrixcolor="#cfe2d4",
          vectorcolor="#fefecd", char_sep_scale=1.8, fontcolor='#444443',
          underline_color='#C2C2C2', ignored_color='#B4B4B4', error_op_color='#A40227',
          ax=None, dpi=200, hush_errors=True) -> PyVizView:
    """
    Parse and evaluate the Python code in the statement string passed in using
    the indicated execution frame. The execution frame of the invoking function
    is used if frame is None.

    The visualization finds the smallest subexpressions that evaluate to
    tensors then underlies them and shows a box or rectangle representing
    the tensor dimensions. Boxes in blue (default) have two or more dimensions
    but rectangles in yellow (default) have one dimension with shape (n,).

    Upon tensor-related execution error, the offending self-expression is
    highlighted (by de-highlighting the other code) and the operator is shown
    using error_op_color.

    To adjust the size of the generated visualization to be smaller or bigger,
    decrease or increase the font size.

    :param statement: A string representing the line of Python code to visualize within an execution frame.
    :param frame: The execution frame in which to evaluate the statement. If None,
                  use the execution frame of the invoking function
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
    :return: Returns a PyVizView holding info about the visualization; from a notebook
             an SVG image will appear. Return none upon parsing error in statement.
    """
    view = PyVizView(statement, fontname, fontsize, dimfontname, dimfontsize, matrixcolor,
                     vectorcolor, char_sep_scale, dpi)

    if frame is None: # use frame of caller if not passed in
        frame = sys._getframe().f_back
    root, tokens = tsensor.parsing.parse(statement, hush_errors=hush_errors)
    if root is None:
        print(f"Can't parse {statement}; root is None")
        # likely syntax error in statement or code I can't handle
        return None
    root_to_viz = root
    try:
        root.eval(frame)
    except tsensor.ast.IncrEvalTrap as e:
        root_to_viz = e.offending_expr
        view.offending_expr = e.offending_expr
        view.cause = e.__cause__
        # Don't raise the exception; keep going to visualize code and erroneous
        # subexpressions. If this function is invoked from clarify() or explain(),
        # the statement will be executed and will fail again during normal execution;
        # an exception will be thrown at that time. Then explain/clarify
        # will update the error message
    subexprs = tsensor.analysis.smallest_matrix_subexpr(root_to_viz)

    # print(statement) # For debugging
    # for i in range(8):
    #     for j in range(10):
    #         print(j,end='')
    # print()

    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=dpi)
    else:
        fig = ax.figure
    view.fignumber = fig.number # track this so that we can determine if the figure has been closed

    ax.axis("off")

    # First, we need to figure out how wide the visualization components are
    # for each sub expression. If these are wider than the sub expression text,
    # than we need to leave space around the sub expression text
    lpad = np.zeros((len(statement),)) # pad for characters
    rpad = np.zeros((len(statement),))
    maxh = 0
    for sub in subexprs:
        w, h = view.boxsize(sub.value)
        maxh = max(h, maxh)
        nexpr = sub.stop.cstop_idx - sub.start.cstart_idx
        if (sub.start.cstart_idx-1)>0 and statement[sub.start.cstart_idx - 1]== ' ': # if char to left is space
            nexpr += 1
        if sub.stop.cstop_idx<len(statement) and statement[sub.stop.cstop_idx]== ' ':     # if char to right is space
            nexpr += 1
        if w>view.wchar * nexpr:
            lpad[sub.start.cstart_idx] += (w - view.wchar) / 2
            rpad[sub.stop.cstop_idx - 1] += (w - view.wchar) / 2

    # Now we know how to place all the elements, since we know what the maximum height is
    view.set_locations(maxh)

    # Find each character's position based upon width of a character and any padding
    charx = np.empty((len(statement),))
    x = view.leftedge
    for i,c in enumerate(statement):
        x += lpad[i]
        charx[i] = x
        x += view.wchar
        x += rpad[i]

    # Draw text for statement or expression
    if view.offending_expr is not None: # highlight erroneous subexpr
        highlight = np.full(shape=(len(statement),), fill_value=False, dtype=bool)
        for tok in tokens[root_to_viz.start.index:root_to_viz.stop.index+1]:
            highlight[tok.cstart_idx:tok.cstop_idx] = True
        errors = np.full(shape=(len(statement),), fill_value=False, dtype=bool)
        for tok in root_to_viz.optokens:
            errors[tok.cstart_idx:tok.cstop_idx] = True
        for i, c in enumerate(statement):
            color = ignored_color
            if highlight[i]:
                color = fontcolor
            if errors[i]: # override color if operator token
                color = error_op_color
            ax.text(charx[i], view.texty, c, color=color, fontname=fontname, fontsize=fontsize)
    else:
        for i, c in enumerate(statement):
            ax.text(charx[i], view.texty, c, color=fontcolor, fontname=fontname, fontsize=fontsize)

    # Compute the left and right edges of subexpressions (alter nodes with info)
    for i,sub in enumerate(subexprs):
        a = charx[sub.start.cstart_idx]
        b = charx[sub.stop.cstop_idx - 1] + view.wchar
        sub.leftx = a
        sub.rightx = b

    # Draw grey underlines and draw matrices
    for i,sub in enumerate(subexprs):
        a,b = sub.leftx, sub.rightx
        pad = view.wchar*0.1
        ax.plot([a-pad, b+pad], [view.liney,view.liney], '-', linewidth=.5, c=underline_color)
        view.draw(ax, sub)

    fig_width = charx[-1] + view.wchar + rpad[-1]
    fig_width_inches = (fig_width) / dpi
    fig_height_inches = view.maxy / dpi
    fig.set_size_inches(fig_width_inches, fig_height_inches)

    ax.set_xlim(0, (fig_width))
    ax.set_ylim(0, view.maxy)

    return view


# ---------------- SHOW AST STUFF ---------------------------

class QuietGraphvizWrapper(graphviz.Source):
    def __init__(self, dotsrc):
        super().__init__(source=dotsrc)

    def _repr_svg_(self):
        return self.pipe(format='svg', quiet=True).decode(self._encoding)

    def savefig(self, filename):
        path = Path(filename)
        if not path.parent.exists:
            os.makedirs(path.parent)

        dotfilename = self.save(directory=path.parent.as_posix(), filename=path.stem)
        format = path.suffix[1:]  # ".svg" -> "svg" etc...
        cmd = ["dot", f"-T{format}", "-o", filename, dotfilename]
        # print(' '.join(cmd))
        graphviz.backend.run(cmd, capture_output=True, check=True, quiet=False)


def astviz(statement:str, frame='current') -> graphviz.Source:
    """
    Display the abstract syntax tree (AST) for the indicated Python code
    in statement. Evaluate that code in the context of frame. If the frame
    is not specified, the default is to execute the code within the context of
    the invoking code. Pass in frame=None to avoid evaluation and just display
    the AST.

    Returns a QuietGraphvizWrapper that renders as SVG in a notebook but
    you can also call `savefig()` to save the file and in a variety of formats,
    according to the file extension.
    """
    return QuietGraphvizWrapper(astviz_dot(statement, frame))


def astviz_dot(statement:str, frame='current') -> str:
    def internal_label(node,color="yellow"):
        text = ''.join(str(t) for t in node.optokens)
        sh = tsensor.analysis._shape(node.value)
        if sh is None:
            return f'<font face="{fontname}" color="#444443" point-size="{fontsize}">{text}</font>'

        sz = 'x'.join([PyVizView.nabbrev(sh[i]) for i in range(len(sh))])
        return f"""<font face="Consolas" color="#444443" point-size="{fontsize}">{text}</font><br/><font face="Arial" color="#444443" point-size="{dimfontsize}">{sz}</font>"""

    root, tokens = tsensor.parsing.parse(statement)

    if frame=='current': # use frame of caller if nothing passed in
        frame = sys._getframe().f_back
        if frame.f_code.co_name=='astviz':
            frame = frame.f_back

    if frame is not None: # if the passed in None, then don't do the evaluation
        root.eval(frame)

    nodes = tsensor.ast.postorder(root)
    atoms = tsensor.ast.leaves(root)
    atomsS = set(atoms)
    ops = [nd for nd in nodes if nd not in atomsS] # keep order

    gr = """digraph G {
        margin=0;
        nodesep=.01;
        ranksep=.3;
        rankdir=BT;
        ordering=out; # keep order of leaves
    """

    matrixcolor = "#cfe2d4"
    vectorcolor = "#fefecd"
    fontname="Consolas"
    fontsize=12
    dimfontsize = 9
    spread = 0

    # Gen leaf nodes
    for i in range(len(tokens)):
        t = tokens[i]
        if t.type!=token.ENDMARKER:
            nodetext = t.value
            # if ']' in nodetext:
            if nodetext==']':
                nodetext = nodetext.replace(']','&zwnj;]') # &zwnj; is 0-width nonjoiner. ']' by itself is bad for DOT
            label = f'<font face="{fontname}" color="#444443" point-size="{fontsize}">{nodetext}</font>'
            _spread = spread
            if t.type==token.DOT:
                _spread=.1
            elif t.type==token.EQUAL:
                _spread=.25
            elif t.type in tsensor.parsing.ADDOP:
                _spread=.4
            elif t.type in tsensor.parsing.MULOP:
                _spread=.2
            gr += f'leaf{id(t)} [shape=box penwidth=0 margin=.001 width={_spread} label=<{label}>]\n'

    # Make sure leaves are on same level
    gr += f'{{ rank=same; '
    for t in tokens:
        if t.type!=token.ENDMARKER:
            gr += f' leaf{id(t)}'
    gr += '\n}\n'

    # Make sure leaves are left to right by linking
    for i in range(len(tokens) - 2):
        t = tokens[i]
        t2 = tokens[i + 1]
        gr += f'leaf{id(t)} -> leaf{id(t2)} [style=invis];\n'

    # Draw internal ops nodes
    for nd in ops:
        label = internal_label(nd)
        sh = tsensor.analysis._shape(nd.value)
        if sh is None:
            color = ""
        else:
            if len(sh)==1:
                color = f'fillcolor="{vectorcolor}" style=filled'
            else:
                color = f'fillcolor="{matrixcolor}" style=filled'
        gr += f'node{id(nd)} [shape=box {color} penwidth=0 margin=0 width=.25 height=.2 label=<{label}>]\n'

    # Link internal nodes to other nodes or leaves
    for nd in nodes:
        kids = nd.kids
        for sub in kids:
            if sub in atomsS:
                gr += f'node{id(nd)} -> leaf{id(sub.token)} [dir=back, penwidth="0.5", color="#6B6B6B", arrowsize=.3];\n'
            else:
                gr += f'node{id(nd)} -> node{id(sub)} [dir=back, penwidth="0.5", color="#6B6B6B", arrowsize=.3];\n'

    gr += "}\n"
    return gr