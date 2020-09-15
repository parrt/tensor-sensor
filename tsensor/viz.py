import sys
import tempfile
import graphviz
import token
from IPython.display import SVG
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
import tsensor
import tsensor.ast
import tsensor.analysis
import tsensor.parsing


class PyVizView:
    """
    An object that collects relevant information about viewing Python code
    with visual annotations
    """
    def __init__(self,
                 statement, fontsize, fontname, dimfontsize,
                 matrixcolor, vectorcolor, char_sep_scale, dpi):
        self.statement = statement
        self.fontsize = fontsize
        self.fontname = fontname
        self.dimfontsize = dimfontsize
        self.matrixcolor = matrixcolor
        self.vectorcolor = vectorcolor
        self.char_sep_scale = char_sep_scale
        self.dpi = dpi


def pyviz(statement:str, frame=None,
          fontsize=16,
          fontname='Consolas',
          dimfontsize=9,
          matrixcolor="#cfe2d4", vectorcolor="#fefecd",
          char_sep_scale=1.8,
          ax=None,
          figsize=None,
          dpi=200 # must match fig of ax passed in and/or savefig
          ):
    if frame is None: # use frame of caller if not passed in
        frame = sys._getframe().f_back
    root, tokens = tsensor.parsing.parse(statement)
    root.eval(frame)
    subexprs = tsensor.analysis.smallest_matrix_subexpr(root)

    print(statement)
    for i in range(8):
        for j in range(10):
            print(j,end='')
    print()

    view = PyVizView(statement, fontsize, fontname, dimfontsize,
                     matrixcolor, vectorcolor, char_sep_scale, dpi)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.gca()

    ax.axis("off")

    leftedge = 25
    texty = 300
    liney = texty - 50
    maxy = texty + 1.4 * fontsize
    wchar = char_sep_scale*fontsize

    # def charx(i): return leftedge + i * wchar


    # First, we need to figure out how wide the visualization components are
    # for each sub expression. If these are wider than the sub expression text,
    # than we need to leave space around the sub expression text
    lpad = np.zeros((len(statement),)) # pad for characters
    rpad = np.zeros((len(statement),))
    for sub in subexprs:
        w = wshape(sub.value, view)
        if w>wchar:
            lpad[sub.start.start_idx] += (w - wchar) / 2
            rpad[sub.stop.stop_idx-1] += (w - wchar) / 2
    # print(lpad)
    # print(rpad)

    # Find each character's position based upon width of a character and any padding
    charx = np.empty((len(statement),))
    x = leftedge
    for i,c in enumerate(statement):
        x += lpad[i]
        charx[i] = x
        x += wchar
        x += rpad[i]
    # print("charx",charx)

    # Draw text for statement or expression
    for i, c in enumerate(statement):
        ax.text(charx[i], texty, c, fontname=fontname, fontsize=fontsize)

    anyvalue = 1 # why does any value work for y?
    fig_width = charx[-1] + wchar + rpad[-1]
    fig_width_inches = (fig_width) / dpi
    fig_height_inches = maxy / dpi
    fig.set_size_inches(fig_width_inches, fig_height_inches)

    ax.set_xlim(0, (fig_width))
    # ax.spines['left'].set_bounds(0, 1)
    ax.set_ylim(0, maxy)

    # Compute the left and right edges of subexpressions (alter nodes with info)
    for i,sub in enumerate(subexprs):
        a = charx[sub.start.start_idx] - lpad[sub.start.start_idx]
        b = charx[sub.stop.stop_idx-1] + wchar + rpad[sub.stop.stop_idx-1]
        sub.leftx = a
        sub.rightx = b

    # Draw grey underlines
    for i,sub in enumerate(subexprs):
        a,b = sub.leftx, sub.rightx
        mid = (a + b) / 2
        ax.plot([a, b], [liney,liney], '-', linewidth=1, c='grey')
        print(sub, sub.start.start_idx, ':', sub.stop.stop_idx, "plot at", a, b, "mid", mid)
        draw(ax, sub, view)


def wmatrix(sh, view): return 150 # width of matrix
def wvector(sh, view): return 400 # flat vector width
def wshape(v, view):
    sh = tsensor.analysis._shape(v)
    if sh is None: return 0
    if len(sh)==1: return wvector(sh, view)
    return wmatrix(sh, view)


def draw(ax, sub, view):
    sh = tsensor.analysis._shape(sub.value)
    if len(sh)==1: draw_vector(ax, sub, view)
    elif len(sh)==2: draw_matrix2D(ax, sub, view)
    elif len(sh)>2: draw_matrix(ax, sub, view)


def draw_vector(ax,sub,view):
    a, b = sub.leftx, sub.rightx
    mid = (a + b) / 2
    width = wshape(sub.value, view)
    rect1 = patches.Rectangle(xy=(mid - width / 2, 75),
                              width=width,
                              height=100,
                              facecolor=view.vectorcolor,
                              edgecolor='grey',
                              fill=True)
    ax.add_patch(rect1)
    sh = tsensor.analysis._shape(sub.value)
    ax.text(mid, 75 + 100, sh[0], horizontalalignment='center', fontsize=view.dimfontsize)


def draw_matrix2D(ax,sub,view):
    a, b = sub.leftx, sub.rightx
    mid = (a + b) / 2
    width = wshape(sub.value, view)
    rect1 = patches.Rectangle(xy=(mid - width / 2, 75),
                              width=width,
                              height=100,
                              facecolor=view.matrixcolor,
                              edgecolor='grey',
                              fill=True)
    ax.add_patch(rect1)


def draw_matrix(ax,sub,view):
    pass


# ----------------

class QuietGraphvizWrapper(graphviz.Source):
    def __init__(self, dotsrc):
        super().__init__(source=dotsrc)

    def _repr_svg_(self):
        return self.pipe(format='svg', quiet=True).decode(self._encoding)
def pyviz_old(statement:str, frame=None) -> graphviz.Source:
    if frame is None: # use frame of caller
        frame = sys._getframe().f_back
    return QuietGraphvizWrapper(pyviz_dot(statement, frame))

def pyviz_dot(statement:str, frame,
              matrixcolor="#cfe2d4", vectorcolor="#fefecd",
              gtype="digraph", gname="G") -> str:

    def elem_label(token_or_node):
        x = tok2node[token_or_node] if token_or_node in tok2node else token_or_node
        sh = tsensor.analysis._shape(x.value) # get value for this node in tree
        label = f'<font face="{fontname}" color="#444443" point-size="{fontsize}">{token_or_node}</font>'
        if sh is not None:
            if len(sh) == 1:
                label = matrix_html(sh[0], None, token_or_node.value, fontname=fontname,
                                    fontsize=fontsize, color=vectorcolor)
            elif len(sh) == 2:
                label = matrix_html(sh[0], sh[1], token_or_node.value, fontname=fontname,
                                    fontsize=fontsize, color=matrixcolor)
        # print(x,'has',sh,label)
        return label

    def internal_label(node):
        text = str(node)
        if node.opstr:
            text = node.opstr
        sh = tsensor.ast._shape(node.value) # get value for this node in tree
        label = f'<font face="{fontname}" color="#444443" point-size="{fontsize}">{text}</font>'
        if sh is not None:
            if len(sh) == 1:
                label = matrix_html(sh[0], None, text, fontname=fontname,
                                    fontsize=fontsize, color=vectorcolor)
            elif len(sh) == 2:
                label = matrix_html(sh[0], sh[1], text, fontname=fontname,
                                    fontsize=fontsize, color=matrixcolor)
        # print(x,'has',sh,label)
        return label

    root, tokens = tsensor.parsing.parse(statement)
    root.eval(frame)
    result = root.value

    # p = tsensor.parsing.PyExprParser(statement)
    # root = p.parse()
    # print(root)
    # print(repr(root))
    nodes = tsensor.ast.postorder(root)
    atoms = tsensor.ast.leaves(root)
    atomsS = set(atoms)
    ops = [nd for nd in nodes if nd not in atomsS] # keep order

    # result = root.eval(frame)

    # ignore = set()
    # def foo(t):
    #     # print("walk", t, repr(t), tsensor.analysis._shape(t.value))
    #     if isinstance(t,tsensor.ast.Member):
    #         if tsensor.analysis._shape(t.obj.value) is None:
    #             # print("\tignore", t)
    #             ignore.add(t)
    #     else:
    #         if tsensor.analysis._shape(t.value) is None:
    #             # print("\tignore", t)
    #             ignore.add(t)
    # tsensor.ast.walk(root, post=foo)
    # print("ignore",[str(n) for n in ignore])
    #
    # map tokens to nodes so we can get variable values
    tok2node = {nd.token:nd for nd in atoms}
    # print(tok2node)

    gr = gtype+" "+gname+""" {
        margin=0;
        nodesep=.01;
        ranksep=.3;
        rankdir=BT;
        ordering=out; # keep order of leaves
    """

    fontname="Consolas"
    fontsize=12
    spread = 0

    # Gen leaf nodes
    for i in range(len(tokens)):
        t = tokens[i]
        if t.type!=token.ENDMARKER:
            label = elem_label(t)
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

    # # Draw internal ops nodes
    # for nd in ops:
    #     # x = tok2node[t] if t in tok2node else t
    #     # if isinstance(nd, tsensor.ast.Member):
    #     #     continue
    #     for sub in nd.kids:
    #         if tsensor.ast._shape(sub.value) is None:
    #             continue
    #     label = internal_label(nd)
    #     gr += f'node{id(nd)} [shape=box penwidth=0 margin=0 height=.3 label=<{label}>]\n'
    #     # gr += f'node{id(nd)} [shape=box penwidth=0 height=.3 margin=0 label=<<font face="Consolas" color="#444443" point-size="12">{nd}</font>>]\n'
    #
    # # Link internal nodes to other nodes or leaves
    # for nd in nodes:
    #     kids = nd.kids
    #     # if isinstance(nd, tsensor.ast.Member) and tsensor.ast._shape(nd.obj) is None:
    #     #     continue
    #     # if isinstance(nd, tsensor.ast.Call) and isinstance(nd.kids[0], tsensor.ast.Member):
    #     #     print('ignore', nd.func, kids)
    #     #     kids = kids[1:]
    #     for sub in kids:
    #         if sub in atomsS:
    #             gr += f'node{id(nd)} -> leaf{id(sub.token)} [dir=back, penwidth="0.5", color="#444443", arrowsize=.4];\n'
    #         else:
    #             gr += f'node{id(nd)} -> node{id(sub)} [dir=back, penwidth="0.5", color="#444443", arrowsize=.4];\n'

    gr += "}\n"
    return gr


def astviz(statement:str, frame=None) -> graphviz.Source:
    return QuietGraphvizWrapper(astviz_dot(statement, frame))


def astviz_dot(statement:str, frame=None) -> str:
    def internal_label(node):
        text = str(node)
        if node.opstr:
            text = node.opstr
        sh = tsensor.analysis._shape(node.value)
        if sh is None:
            return f'<font face="{fontname}" color="#444443" point-size="{fontsize}">{text}</font>'

        if len(sh)==1:
            sz = str(sh[0])
        else:
            sz = f"{sh[0]}x{sh[1]}"
        return f"""<font face="Consolas" color="#444443" point-size="{fontsize}">{text}</font><br/><font face="Consolas" color="#444443" point-size="{dimfontsize}">{sz}</font>"""

    root, tokens = tsensor.parsing.parse(statement)
    if frame is not None:
        root.eval(frame)

    nodes = tsensor.ast.postorder(root)
    atoms = tsensor.ast.leaves(root)
    atomsS = set(atoms)
    ops = [nd for nd in nodes if nd not in atomsS] # keep order
    # map tokens to nodes so we can get variable values
    tok2node = {nd.token:nd for nd in atoms}

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
        # for sub in nd.kids:
        #     if tsensor.analysis._shape(sub.value) is None:
        #         continue
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


def matrix_html(nrows, ncols, label, fontsize=12, fontname="Consolas", dimfontsize=9, color="#cfe2d4"):
    isvec = ncols==None
    if isvec:
        sz = str(nrows)
        ncols=nrows
        nrows=1
    else:
        sz = f"{nrows}x{ncols}"
    w = ncols*20
    h = nrows*20
    if ncols==1:
        w = 15
    if nrows==1:
        h = 15
    html = f"""
    <table fixedsize="true" width="{w}" height="{h+2*fontsize*1.1}" BORDER="0" CELLPADDING="0" CELLBORDER="1" CELLSPACING="0">
    <tr>
    <td fixedsize="true" width="{w}" height="{fontsize*1.1}" cellspacing="0" cellpadding="0" border="0" valign="bottom" align="center">
    <font face="{fontname}" color="#444443" point-size="{dimfontsize}">{sz}</font>
    </td>
    </tr>
    <tr>    
    <td fixedsize="true" width="{w}" height="{h}" cellspacing="0" cellpadding="0" bgcolor="{color}" border="1" sides="ltbr" align="center">
    <font face="{fontname}" color="#444443" point-size="{fontsize}">{label}</font>
    </td>
    </tr>
    </table>"""
    return html


