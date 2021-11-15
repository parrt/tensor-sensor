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
import tsensor

# Parse tree definitions
# I found package ast in python3 lib after I built this. whoops. No biggie.
# This tree structure is easier to visit for my purposes here. Also lets me
# control the kinds of statements I process.

class ParseTreeNode:
    def __init__(self, parser):
        self.parser = parser # which parser object created this node;
                             # useful for getting access to the code string from a token
        self.value = None # used during evaluation
        self.start = None # start token
        self.stop = None  # end token
    def eval(self, frame):
        """
        Evaluate the expression represented by this (sub)tree in context of frame.
        Try any exception found while evaluating and remember which operation that
        was in this tree
        """
        try:
            self.value = eval(str(self), frame.f_globals, frame.f_locals)
        except BaseException as e:
            raise IncrEvalTrap(self) from e
        # print(self, "=>", self.value)
        return self.value
    @property
    def optokens(self): # the associated token if atom or representative token if operation
        return None
    @property
    def kids(self):
        return []
    def clarify(self):
        return None
    def __str__(self):
        # Extract text from the original code string using token character indexes
        return self.parser.code[self.start.cstart_idx:self.stop.cstop_idx]
    def __repr__(self):
        fields = self.__dict__.copy()
        kill = ['start', 'stop', 'lbrack', 'lparen', 'parser']
        for name in kill:
            if name in fields: del fields[name]
        args = [
            v + '=' + fields[v].__repr__()
            for v in fields
            if v != 'value' or fields['value'] is not None
        ]
        args = ','.join(args)
        return f"{self.__class__.__name__}({args})"

class Assign(ParseTreeNode):
    def __init__(self, parser, op, lhs, rhs, start, stop):
        super().__init__(parser)
        self.op, self.lhs, self.rhs = op, lhs, rhs
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.value = self.rhs.eval(frame)
        # Don't eval this node as it causes side effect of making actual assignment to lhs
        self.lhs.value = self.value
        return self.value
    @property
    def optokens(self):
        return [self.op]
    @property
    def kids(self):
        return [self.lhs, self.rhs]


class Call(ParseTreeNode):
    def __init__(self, parser, func, lparen, args, start, stop):
        super().__init__(parser)
        self.func = func
        self.lparen = lparen
        self.args = args
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.func.eval(frame)
        for a in self.args:
            a.eval(frame)
        return super().eval(frame)
    def clarify(self):
        arg_msgs = []
        for a in self.args:
            ashape = tsensor.analysis._shape(a.value)
            if ashape:
                arg_msgs.append(f"arg {a} w/shape {ashape}")
        if len(arg_msgs)==0:
            return f"Cause: {self}"
        return f"Cause: {self} tensor " + ', '.join(arg_msgs)
    @property
    def optokens(self):
        f = None # assume complicated like a[i](args) with weird func expr
        if isinstance(self.func, Member):
            f = self.func.member
        elif isinstance(self.func, Atom):
            f = self.func
        if f:
            return [f.token,self.lparen,self.stop]
        return [self.lparen,self.stop]
    @property
    def kids(self):
        return [self.func]+self.args


class Return(ParseTreeNode):
    def __init__(self, parser, result, start, stop):
        super().__init__(parser)
        self.result = result
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.value = [a.eval(frame) for a in self.result]
        if len(self.value)==1:
            self.value = self.value[0]
        return self.value
    @property
    def optokens(self):
        return [self.start]
    @property
    def kids(self):
        return self.result


class Index(ParseTreeNode):
    def __init__(self, parser, arr, lbrack, index, start, stop):
        super().__init__(parser)
        self.arr = arr
        self.lbrack = lbrack
        self.index = index
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.arr.eval(frame)
        for i in self.index:
            i.eval(frame)
        return super().eval(frame)
    @property
    def optokens(self):
        return [self.lbrack,self.stop]
    @property
    def kids(self):
        return [self.arr] + self.index


class Member(ParseTreeNode):
    def __init__(self, parser, op, obj, member, start, stop):
        super().__init__(parser)
        self.op = op # always DOT
        self.obj = obj
        self.member = member
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.obj.eval(frame)
        # don't eval member as it's just a name to look up in obj
        return super().eval(frame)
    @property
    def optokens(self): # the associated token if atom or representative token if operation
        return [self.op]
    @property
    def kids(self):
        return [self.obj, self.member]


class BinaryOp(ParseTreeNode):
    def __init__(self, parser, op, lhs, rhs, start, stop):
        super().__init__(parser)
        self.op, self.lhs, self.rhs = op, lhs, rhs
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.lhs.eval(frame)
        self.rhs.eval(frame)
        return super().eval(frame)
    def clarify(self):
        opnd_msgs = []
        lshape = tsensor.analysis._shape(self.lhs.value)
        rshape = tsensor.analysis._shape(self.rhs.value)
        if lshape:
            opnd_msgs.append(f"operand {self.lhs} w/shape {lshape}")
        if rshape:
            opnd_msgs.append(f"operand {self.rhs} w/shape {rshape}")
        return f"Cause: {self.op} on tensor " + ' and '.join(opnd_msgs)
    @property
    def optokens(self): # the associated token if atom or representative token if operation
        return [self.op]
    @property
    def kids(self):
        return [self.lhs, self.rhs]


class UnaryOp(ParseTreeNode):
    def __init__(self, parser, op, opnd, start, stop):
        super().__init__(parser)
        self.op = op
        self.opnd = opnd
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.opnd.eval(frame)
        return super().eval(frame)
    @property
    def optokens(self):
        return [self.op]
    @property
    def kids(self):
        return [self.opnd]


class ListLiteral(ParseTreeNode):
    def __init__(self, parser, elems, start, stop):
        super().__init__(parser)
        self.elems = elems
        self.start, self.stop = start, stop
    def eval(self, frame):
        for i in self.elems:
            i.eval(frame)
        return super().eval(frame)
    @property
    def kids(self):
        return self.elems


class TupleLiteral(ParseTreeNode):
    def __init__(self, parser, elems, start, stop):
        super().__init__(parser)
        self.elems = elems
        self.start, self.stop = start, stop
    def eval(self, frame):
        for i in self.elems:
            i.eval(frame)
        return super().eval(frame)
    @property
    def kids(self):
        return self.elems


class SubExpr(ParseTreeNode):
    # record parens for later display to keep precedence
    def __init__(self, parser, e, start, stop):
        super().__init__(parser)
        self.e = e
        self.start, self.stop = start, stop
    def eval(self, frame):
        self.value = self.e.eval(frame)
        return self.value # don't re-evaluate
    @property
    def optokens(self):
        return [self.start, self.stop]
    @property
    def kids(self):
        return [self.e]


class Atom(ParseTreeNode):
    def __init__(self, parser, token):
        super().__init__(parser)
        self.token = token
        self.start, self.stop = token, token
    def eval(self, frame):
        if self.token.type == tsensor.parsing.COLON:
            return ':' # fake a value here
        return super().eval(frame)
    def __repr__(self):
        # v = f"{{{self.value}}}" if hasattr(self,'value') and self.value is not None else ""
        return self.token.value


def postorder(t):
    nodes = []
    _postorder(t, nodes)
    return nodes


def _postorder(t, nodes):
    if t is None:
        return
    for sub in t.kids:
        _postorder(sub, nodes)
    nodes.append(t)


def leaves(t):
    nodes = []
    _leaves(t, nodes)
    return nodes


def _leaves(t, nodes):
    if t is None:
        return
    if len(t.kids) == 0:
        nodes.append(t)
        return
    for sub in t.kids:
        _leaves(sub, nodes)


def walk(t, pre=lambda x: None, post=lambda x: None):
    if t is None:
        return
    pre(t)
    for sub in t.kids:
        walk(sub, pre, post)
    post(t)


class IncrEvalTrap(BaseException):
    """
    Used during re-evaluation of python line that threw exception to trap which
    subexpression caused the problem.
    """
    def __init__(self, offending_expr):
        self.offending_expr = offending_expr # where in tree did we get exception?
