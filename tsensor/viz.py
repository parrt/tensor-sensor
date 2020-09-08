import sys
import graphviz
import token

import tsensor.ast
import tsensor.explain
import tsensor.parse

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


def pyviz_graphviz(statement, frame,
                   matrixcolor="#cfe2d4", vectorcolor="#fefecd",
                   gtype="digraph", gname="G"):

    def elem_label(token_or_node):
        x = tok2node[token_or_node] if token_or_node in tok2node else token_or_node
        sh = tsensor.ast._shape(x.value) # get value for this node in tree
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

    p = tsensor.parse.PyExprParser(statement)
    root = p.parse()
    print(root)
    nodes = tsensor.ast.postorder(root)
    atoms = tsensor.ast.leaves(root)
    atomsS = set(atoms)
    ops = [nd for nd in nodes if nd not in atomsS] # keep order

    # map tokens to nodes so we can get variable values
    tok2node = {nd.token:nd for nd in atoms}
    print(tok2node)

    result = root.eval(frame)

    gr = gtype+" "+gname+""" {
        nodesep=.0;
        ranksep=.3;
        rankdir=BT;
        ordering=out; # keep order of leaves
    """

    fontname="Consolas"
    fontsize=12
    spread = .2

    # Gen leaf nodes
    for i in range(len(p.tokens)):
        t = p.tokens[i]
        if t.type!=token.ENDMARKER:
            label = elem_label(t)
            if t.type==token.DOT:
                spread=.1
            if t.type==token.EQUAL:
                spread=.25
            if t.type in tsensor.parse.ADDOP:
                spread=.5
            if t.type in tsensor.parse.MULOP:
                spread=.2
            gr += f'leaf{id(t)} [shape=box penwidth=0 margin=.001 width={spread} label=<{label}>]\n'

    # Make sure leaves are on same level
    gr += f'{{ rank=same; '
    for t in p.tokens:
        if t.type!=token.ENDMARKER:
            gr += f' leaf{id(t)}'
    gr += '\n}\n'

    # Make sure leaves are left to right by linking
    for i in range(len(p.tokens) - 2):
        t = p.tokens[i]
        t2 = p.tokens[i + 1]
        gr += f'leaf{id(t)} -> leaf{id(t2)} [style=invis];\n'

    # Draw internal ops nodes
    for nd in ops:
        # x = tok2node[t] if t in tok2node else t
        label = internal_label(nd)
        gr += f'node{id(nd)} [shape=box penwidth=0 margin=0 height=.3 label=<{label}>]\n'
        # gr += f'node{id(nd)} [shape=box penwidth=0 height=.3 margin=0 label=<<font face="Consolas" color="#444443" point-size="12">{nd}</font>>]\n'

    # Link internal nodes to other nodes or leaves
    for nd in nodes:
        kids = nd.kids
        if isinstance(nd, tsensor.ast.Member):
            continue
        if isinstance(nd, tsensor.ast.Call) and isinstance(nd.kids[0], tsensor.ast.Member):
            print('ignore', nd.func, kids)
            kids = kids[1:]
        for sub in kids:
            if sub in atomsS:
                gr += f'node{id(nd)} -> leaf{id(sub.token)} [dir=back, penwidth="0.5", color="#444443", arrowsize=.4];\n'
            else:
                gr += f'node{id(nd)} -> node{id(sub)} [dir=back, penwidth="0.5", color="#444443", arrowsize=.4];\n'

    gr += "}\n"
    return gr


import torch
W = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([9, 10]).reshape(2, 1)
x = torch.tensor([4, 5]).reshape(2, 1)
h = torch.tensor([1, 2])
a = 3

frame = sys._getframe()
html1 = pyviz_graphviz("b = W@b + h.dot(h) + torch.abs(torch.tensor(34))", frame)
# html2 = pyviz_html("b = W.T@W")

#html = f"digraph foo {{ compound=true; rankdir=TB;\n {html1} {html2} G1 -> G2 }}"

# graphviz.Source(html1).view()
graphviz.Source(html1).view()