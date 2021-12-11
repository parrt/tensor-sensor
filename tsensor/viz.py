"""
MIT License

Copyright (c) 2021 Terence Parr

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
import matplotlib.colors as mc
from IPython.display import display, SVG
from IPython import get_ipython

import numpy as np
import tsensor
import tsensor.ast
import tsensor.analysis
import tsensor.parsing


class DTypeColorInfo:
    """
    Track the colors for various types, the transparency range, and bit precisions.
    By default, green indicates floating-point, blue indicates integer, and orange
    indicates complex numbers. The more saturated the color (lower transparency),
    the higher the precision.
    """
    orangeish = '#FDD66C'
    limeish = '#A8E1B0'
    blueish = '#7FA4D3'
    grey = '#EFEFF0'
    default_dtype_colors = {'float': limeish, 'int': blueish, 'complex': orangeish, 'other': grey}
    default_dtype_precisions = [32, 64, 128]  # hard to see diff if we use [4, 8, 16, 32, 64, 128]
    default_dtype_alpha_range = (0.5, 1.0)    # use (0.1, 1.0) if more precision values

    def __init__(self, dtype_colors=None, dtype_precisions=None, dtype_alpha_range=None):
        if dtype_colors is None:
            dtype_colors = DTypeColorInfo.default_dtype_colors
        if dtype_precisions is None:
            dtype_precisions = DTypeColorInfo.default_dtype_precisions
        if dtype_alpha_range is None:
            dtype_alpha_range = DTypeColorInfo.default_dtype_alpha_range

        if not isinstance(dtype_colors, dict) or (len(dtype_colors) > 0 and \
           not isinstance(list(dtype_colors.values())[0], str)):
            raise TypeError(
                "dtype_colors should be a dict mapping type name to color name or color hex RGB values."
            )

        self.dtype_colors, self.dtype_precisions, self.dtype_alpha_range = \
            dtype_colors, dtype_precisions, dtype_alpha_range

    def color(self, dtype):
        """Get color based on type and precision. Return list of RGB and alpha"""
        dtype_name, dtype_precision = PyVizView._split_dtype_precision(dtype)
        if dtype_name not in self.dtype_colors:
            return self.dtype_colors['other']
        color = self.dtype_colors[dtype_name]
        dtype_precision = int(dtype_precision)
        if dtype_precision not in self.dtype_precisions:
            return self.dtype_colors['other']

        color = mc.hex2color(color) if color[0] == '#' else mc.cnames[color]
        precision_idx = self.dtype_precisions.index(dtype_precision)
        nshades = len(self.dtype_precisions)
        alphas = np.linspace(*self.dtype_alpha_range, nshades)
        alpha = alphas[precision_idx]
        return list(color) + [alpha]


class PyVizView:
    """
    An object that collects relevant information about viewing Python code
    with visual annotations.
    """
    def __init__(self, statement, fontname, fontsize, dimfontname, dimfontsize,
                 char_sep_scale, dpi,
                 dtype_colors=None, dtype_precisions=None, dtype_alpha_range=None):
        self.statement = statement
        self.fontsize = fontsize
        self.fontname = fontname
        self.dimfontsize = dimfontsize
        self.dimfontname = dimfontname
        self.char_sep_scale = char_sep_scale
        self.dpi = dpi
        self.dtype_color_info = DTypeColorInfo(dtype_colors, dtype_precisions, dtype_alpha_range)
        self._dtype_encountered = set() # which types, like 'int64', did we find in one plot?
        self.wchar = self.char_sep_scale * self.fontsize
        self.wchar_small = self.char_sep_scale * (self.fontsize - 2)  # for <int32> typenames
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

    @staticmethod
    def _split_dtype_precision(s):
        """Split the final integer part from a string"""
        head = s.rstrip('0123456789')
        tail = s[len(head):]
        return head, tail

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
        box2line = line2text*2.6
        self.texty = self.bottomedge + maxh + box2line + line2text
        self.liney = self.bottomedge + maxh + box2line
        self.box_topy = self.bottomedge + maxh
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
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight', pad_inches=0)
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
        ty = tsensor.analysis._dtype(v)
        if sh is None: return None
        if len(sh)==1: return self.vector_size(sh, ty)
        return self.matrix_size(sh, ty)

    def matrix_size(self, sh, ty):
        """
        How wide and tall should we draw the box representing a matrix.
        """
        if len(sh)==1 and sh[0]==1:
            return self.vector_size(sh, ty)

        if len(sh) > 1 and sh[0] == 1 and sh[1] == 1:
            # A special case where we have a 1x1 matrix extending into the screen.
            # Make the 1x1 part a little bit wider than a vector so it's more readable
            w, h = 2 * self.vector_size_scaler * self.wchar, 2 * self.vector_size_scaler * self.wchar
        elif len(sh) > 1 and sh[1] == 1:
            w, h = self.vector_size_scaler * self.wchar, self.matrix_size_scaler * self.wchar
        elif len(sh)>1 and sh[0]==1:
            w, h = self.matrix_size_scaler * self.wchar, self.vector_size_scaler * self.wchar
        else:
            w, h = self.matrix_size_scaler * self.wchar, self.matrix_size_scaler * self.wchar
        return w, h

    def vector_size(self, sh, ty):
        """
        How wide and tall is a vector?  It's not a function of vector length; instead
        we make a row vector with same width as a matrix but height of just one char.
        For consistency with matrix_size(), I pass in shape, though it's ignored.
        """
        return self.matrix_size_scaler * self.wchar, self.vector_size_scaler * self.wchar

    def draw(self, ax, sub):
        sh = tsensor.analysis._shape(sub.value)
        ty = tsensor.analysis._dtype(sub.value)
        self._dtype_encountered.add(ty)
        if len(sh) == 1:
            self.draw_vector(ax, sub, sh, ty)
        else:
            self.draw_matrix(ax, sub, sh, ty)

    def draw_vector(self,ax,sub, sh, ty: str):
        mid = (sub.leftx + sub.rightx) / 2
        w,h = self.vector_size(sh, ty)
        color = self.dtype_color_info.color(ty)
        rect1 = patches.Rectangle(xy=(mid - w/2, self.box_topy-h),
                                  width=w,
                                  height=h,
                                  linewidth=self.linewidth,
                                  facecolor=color,
                                  edgecolor='grey',
                                  fill=True)
        ax.add_patch(rect1)

        # Text above vector rectangle
        ax.text(mid, self.box_topy + self.dim_ypadding, self.nabbrev(sh[0]),
                horizontalalignment='center',
                fontname=self.dimfontname, fontsize=self.dimfontsize)
        # Type info at the bottom of everything
        ax.text(mid, self.box_topy - self.hchar, '<${\mathit{'+ty+'}}$>',
                verticalalignment='top', horizontalalignment='center',
                fontname=self.dimfontname, fontsize=self.dimfontsize-2)

    def draw_matrix(self,ax,sub, sh, ty):
        mid = (sub.leftx + sub.rightx) / 2
        w,h = self.matrix_size(sh, ty)
        box_left = mid - w / 2
        color = self.dtype_color_info.color(ty)

        if len(sh) > 2:
            back_rect = patches.Rectangle(xy=(box_left + self.shift3D, self.box_topy - h + self.shift3D),
                                          width=w,
                                          height=h,
                                          linewidth=self.linewidth,
                                          facecolor=color,
                                          edgecolor='grey',
                                          fill=True)
            ax.add_patch(back_rect)
        rect = patches.Rectangle(xy=(box_left, self.box_topy - h),
                                  width=w,
                                  height=h,
                                  linewidth=self.linewidth,
                                  facecolor=color,
                                  edgecolor='grey',
                                  fill=True)
        ax.add_patch(rect)

        # Text above matrix rectangle
        ax.text(box_left, self.box_topy - h/2, self.nabbrev(sh[0]),
                verticalalignment='center', horizontalalignment='right',
                fontname=self.dimfontname, fontsize=self.dimfontsize, rotation=90)

        # Note: this was always true since matrix...
        textx = mid
        texty = self.box_topy + self.dim_ypadding
        if len(sh) > 2:
            texty += self.dim_ypadding
            textx += self.shift3D

        # Text to the left
        ax.text(textx, texty, self.nabbrev(sh[1]), horizontalalignment='center',
                fontname=self.dimfontname, fontsize=self.dimfontsize)

        if len(sh) > 2:
            # Text to the right
            ax.text(box_left+w, self.box_topy - h/2, self.nabbrev(sh[2]),
                    verticalalignment='center', horizontalalignment='center',
                    fontname=self.dimfontname, fontsize=self.dimfontsize,
                    rotation=45)

        bottom_text_line = self.box_topy - h - self.dim_ypadding
        if len(sh) > 3:
            # Text below
            remaining = r"$\cdots\mathsf{x}$"+r"$\mathsf{x}$".join([self.nabbrev(sh[i]) for i in range(3,len(sh))])
            bottom_text_line = self.box_topy - h - self.dim_ypadding
            ax.text(mid, bottom_text_line, remaining,
                    verticalalignment='top', horizontalalignment='center',
                    fontname=self.dimfontname, fontsize=self.dimfontsize)
            bottom_text_line -= self.hchar + self.dim_ypadding

        # Type info at the bottom of everything
        ax.text(mid, bottom_text_line, '<${\mathit{'+ty+'}}$>',
                verticalalignment='top', horizontalalignment='center',
                fontname=self.dimfontname, fontsize=self.dimfontsize-2)

    @staticmethod
    def nabbrev(n: int) -> str:
        if n % 1_000_000 == 0:
            return str(n // 1_000_000)+'m'
        if n % 1_000 == 0:
            return str(n // 1_000)+'k'
        return str(n)


def pyviz(statement: str, frame=None,
          fontname='Consolas', fontsize=13,
          dimfontname='Arial', dimfontsize=9, char_sep_scale=1.8, fontcolor='#444443',
          underline_color='#C2C2C2', ignored_color='#B4B4B4', error_op_color='#A40227',
          ax=None, dpi=200, hush_errors=True,
          dtype_colors=None, dtype_precisions=None, dtype_alpha_range=None) -> PyVizView:
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
    :param char_sep_scale: It is notoriously difficult to discover how wide and tall
                           text is when plotted in matplotlib. In fact there's probably,
                           no hope to discover this information accurately in all cases.
                           Certainly, I gave up after spending huge effort. We have a
                           situation here where the font should be constant width, so
                           we can just use a simple scalar times the font size to get
                           a reasonable approximation of the width and height of a
                           character box; the default of 1.8 seems to work reasonably
                           well for a wide range of fonts, but you might have to tweak it
                           when you change the font size.
    :param fontcolor:  The color of the Python code.
    :param underline_color:  The color of the lines that underscore tensor subexpressions; default is grey
    :param ignored_color: The de-highlighted color for de-emphasizing code not involved in an erroneous sub expression
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
    :param dtype_colors: map from dtype w/o precision like 'int' to color
    :param dtype_precisions: list of bit precisions to colorize, such as [32,64,128]
    :param dtype_alpha_range: all tensors of the same type are drawn to the same color,
                              and the alpha channel is used to show precision; the
                              smaller the bit size, the lower the alpha channel. You
                              can play with the range to get better visual dynamic range
                              depending on how many precisions you want to display.
    :return: Returns a PyVizView holding info about the visualization; from a notebook
             an SVG image will appear. Return none upon parsing error in statement.
    """
    view = PyVizView(statement, fontname, fontsize, dimfontname, dimfontsize, char_sep_scale, dpi,
                     dtype_colors, dtype_precisions, dtype_alpha_range)

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
        # update width to include horizontal room for type text like int32
        ty = tsensor.analysis._dtype(sub.value)
        w_typename = len(ty) * view.wchar_small
        w = max(w, w_typename)
        maxh = max(h, maxh)
        nexpr = sub.stop.cstop_idx - sub.start.cstart_idx
        if (sub.start.cstart_idx-1)>0 and statement[sub.start.cstart_idx - 1]== ' ':  # if char to left is space
            nexpr += 1
        if sub.stop.cstop_idx<len(statement) and statement[sub.stop.cstop_idx]== ' ': # if char to right is space
            nexpr += 1
        if w > view.wchar * nexpr:
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
    fig_width_inches = fig_width / dpi
    fig_height_inches = view.maxy / dpi
    fig.set_size_inches(fig_width_inches, fig_height_inches)

    ax.set_xlim(0, fig_width)
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
        path.parent.mkdir(exist_ok=True)

        dotfilename = self.save(directory=path.parent.as_posix(), filename=path.stem)
        format = path.suffix[1:]  # ".svg" -> "svg" etc...
        cmd = ["dot", f"-T{format}", "-o", filename, dotfilename]
        # print(' '.join(cmd))
        if graphviz.__version__ <= '0.17':
            graphviz.backend.run(cmd, capture_output=True, check=True, quiet=False)
        else:
            graphviz.backend.execute.run_check(cmd, capture_output=True, check=True, quiet=False)


def astviz(statement:str, frame='current',
           dtype_colors=None, dtype_precisions=None, dtype_alpha_range=None) -> graphviz.Source:
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
    return QuietGraphvizWrapper(
        astviz_dot(statement, frame,
                   dtype_colors, dtype_precisions, dtype_alpha_range)
    )


def astviz_dot(statement:str, frame='current',
               dtype_colors=None, dtype_precisions=None, dtype_alpha_range=None) -> str:
    def internal_label(node):
        sh = tsensor.analysis._shape(node.value)
        ty = tsensor.analysis._dtype(node.value)
        text = ''.join(str(t) for t in node.optokens)
        if sh is None:
            return f'<font face="{fontname}" point-size="{fontsize}">{text}</font>'

        sz = 'x'.join([PyVizView.nabbrev(sh[i]) for i in range(len(sh))])
        return f"""<font face="Consolas" color="#444443" point-size="{fontsize}">{text}</font><br/><font face="Arial" color="#444443" point-size="{dimfontsize}">{sz}</font><br/><font face="Arial" color="#444443" point-size="{dimfontsize}">&lt;{ty}&gt;</font>"""

    dtype_color_info = DTypeColorInfo(dtype_colors, dtype_precisions, dtype_alpha_range)

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
            ty = tsensor.analysis._dtype(nd.value)
            color = dtype_color_info.color(ty)
            color = mc.rgb2hex(color, keep_alpha=True)
            color = f'fillcolor="{color}" style=filled'
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
