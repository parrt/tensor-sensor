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
import inspect
import sys
from io import BytesIO
import traceback
import torch
import token
from tokenize import tokenize,\
    NUMBER, STRING, NAME, OP, ENDMARKER, LPAR, LSQB, RPAR, RSQB, EQUAL, COMMA, COLON,\
    PLUS, MINUS, STAR, SLASH, AT, PERCENT, TILDE, DOT


"""
The goal of this library is to generate more helpful exception
messages for numpy/pytorch tensor algebra expressions.  Because the
matrix algebra in pytorch and numpy are all done in C/C++, they do not
have access to the Python execution environment so they are literally
unable to give information about which Python variables caused the
problem.  Only by catching the exception and then analyzing the Python
code can we get this kind of an error message.

Imagine you have a complicated little matrix expression like:

W @ torch.dot(b,b)+ torch.eye(2,2)@x + z

And you get this unhelpful error message from pytorch:

RuntimeError: 1D tensors expected, got 2D, 2D tensors at [...]/THTensorEvenMoreMath.cpp:83

There are two problems: it does not tell you which of the sub
expressions threw the exception and it does not tell you what the
shape of relevant operands are.  This library that lets you
do this:

import tsensor
with tsensor.dbg():
    W @ torch.dot(b,b)+ torch.eye(2,2)@x + z

which then emits the following better error message:

Call torch.dot(b,b) has arg b w/shape [2, 1], arg b w/shape [2, 1]

The with statement allows me to trap exceptions that occur and then I
literally parse the Python code of the offending line, build an
expression tree, and then incrementally evaluate the operands
bottom-up until I run into an exception. That tells me which of the
subexpressions caused the problem and then I can pull it apart and
ask if any of those operands are matrices.

Here’s another default error message that is almost helpful for expression W @ z:

RuntimeError: size mismatch, get 2, 2x2,3

But this library gives:

Operation @ has operand W w/shape torch.Size([2, 2]) and operand z w/shape torch.Size([3])
"""

ADDOP     = {PLUS, MINUS}
MULOP     = {STAR, SLASH, AT, PERCENT}
UNARYOP   = {TILDE}


class Token:
    def __init__(self, type, value):
        self.type, self.value = type, value
    def __repr__(self):
        return f"<{token.tok_name[self.type]}:{self.value}>"
    def __str__(self):
        return self.value


class IncrEvalTrap(BaseException):
    """
    Used during re-evaluation of python line that threw exception to trap which
    subexpression caused the problem.
    """
    def __init__(self, offending_expr):
        self.offending_expr = offending_expr # where in tree did we get exception?


# Parse tree definitions
# I found package ast in python3 lib after I built this. whoops. No biggie.
# This tree structure is easier to visit for my purposes here.

class ParseTreeNode:
    def __init__(self):
        self.value = None # used during evaluation
    def eval(self, frame):
        """
        Evaluate the expression represented by this (sub)tree in context of frame.
        Try any exception found while evaluating and remember which operation that
        was in this tree
        """
        try:
            self.value = eval(str(self), frame.f_locals, frame.f_globals)
        except:
            raise IncrEvalTrap(self)
        # print(self, "=>", self.value)
        return self.value
    @property
    def left(self): return None
    @property
    def right(self): return None
    def explain(self):
        return None
    def __str__(self):
        pass
    def __repr__(self):
        args = [v+'='+self.__dict__[v].__repr__() for v in self.__dict__ if v!='value' or self.__dict__['value'] is not None]
        args = ','.join(args)
        return f"{self.__class__.__name__}({args})"

class Assign(ParseTreeNode):
    def __init__(self, lhs, rhs):
        super().__init__()
        self.lhs, self.rhs = lhs, rhs
    def eval(self, frame):
        "Only consider rhs of assignment where our expr errors will occur"
        self.value = self.rhs.eval(frame)
        return self.value
    @property
    def left(self): return self.lhs
    @property
    def right(self): return self.rhs
    def __str__(self):
        return str(self.lhs)+'='+str(self.rhs)

class Call(ParseTreeNode):
    def __init__(self, name, args):
        super().__init__()
        self.name = name
        self.args = args
    def eval(self, frame):
        for a in self.args:
            a.eval(frame)
        return super().eval(frame)
    def explain(self):
        arg_msgs = []
        for a in self.args:
            ashape = _shape(a.value)
            if ashape:
                arg_msgs.append(f"arg {a} w/shape {ashape}")
        if len(arg_msgs)==0:
            return f"Cause: {self}"
        return f"Cause: {self} tensor " + ', '.join(arg_msgs)
    @property
    def left(self): return self.args
    def __str__(self):
        args_ = ','.join([str(a) for a in self.args])
        return f"{self.name}({args_})"

class Index(ParseTreeNode):
    def __init__(self, name, index):
        super().__init__()
        self.name = name
        self.index = index
    def eval(self, frame):
        for i in self.index:
            i.eval(frame)
        return super().eval(frame)
    @property
    def left(self): return self.index
    def __str__(self):
        i = self.index
        i = ','.join(str(v) for v in i)
        return f"{self.name}[{i}]"

class BinaryOp(ParseTreeNode):
    def __init__(self, op, lhs, rhs):
        super().__init__()
        self.op, self.lhs, self.rhs = op, lhs, rhs
    def eval(self, frame):
        self.lhs.eval(frame)
        self.rhs.eval(frame)
        return super().eval(frame)
    def explain(self):
        opnd_msgs = []
        lshape = _shape(self.lhs.value)
        rshape = _shape(self.rhs.value)
        if lshape:
            opnd_msgs.append(f"operand {self.lhs} w/shape {lshape}")
        if rshape:
            opnd_msgs.append(f"operand {self.rhs} w/shape {rshape}")
        return f"Cause: {self.op} on tensor " + ' and '.join(opnd_msgs)
    @property
    def left(self): return self.lhs
    @property
    def right(self): return self.rhs
    def __str__(self):
        return f"{self.lhs}{self.op}{self.rhs}"

class UnaryOp(ParseTreeNode):
    def __init__(self, op, opnd):
        super().__init__()
        self.op = op
        self.opnd = opnd
    def eval(self, frame):
        self.opnd.eval(frame)
        return super().eval(frame)
    @property
    def left(self): return self.opnd
    def __str__(self):
        return f"{self.op}{self.opnd}"

class ListLiteral(ParseTreeNode):
    def __init__(self, elems):
        super().__init__()
        self.elems = elems
    def eval(self, frame):
        for i in self.elems:
            i.eval(frame)
        return super().eval(frame)
    @property
    def left(self): return self.elems
    def __str__(self):
        if isinstance(self.elems,list):
            elems_ = ','.join(str(e) for e in self.elems)
        else:
            elems_ = self.elems
        return f"[{elems_}]"

class SubExpr(ParseTreeNode):
    # record parens for later display to keep precedence
    def __init__(self, e):
        super().__init__()
        self.e = e
    def eval(self, frame):
        self.e.eval(frame)
        return self.e.value # don't re-evaluate
    @property
    def left(self): return self.e
    def __str__(self):
        return f"({self.e})"

class Atom(ParseTreeNode):
    def __init__(self, nametok):
        super().__init__()
        self.nametok = nametok
    def __repr__(self):
        v = f"{{{self.value}}}" if hasattr(self,'value') and self.value is not None else ""
        return self.nametok.value+v
    def __str__(self):
        return self.nametok.value

class String(ParseTreeNode):
    def __init__(self, stok):
        super().__init__()
        self.stok = stok
    def __repr__(self):
        return f"'{self.stok.value}'"
    def __str__(self):
        return f"'{self.stok.value}'"


class PyExprParser:
    def __init__(self, code):
        self.code = code
        self.tokens = mytokenize(code)
        self.t = 0 # current lookahead

    def parse(self):
        # print("\nparse", self.code)
        # print(self.tokens)
        s = self.statement()
        self.match(ENDMARKER)
        return s

    def statement(self):
        lhs = self.expression()
        rhs = None
        if self.LA(1) == EQUAL:
            self.t += 1
            rhs = self.expression()
            return Assign(lhs,rhs)
        return lhs

    def expression(self):
        return self.addexpr()

    def addexpr(self):
        root = self.multexpr()
        while self.LA(1) in ADDOP:
            op = self.LT(1)
            self.t += 1
            b = self.multexpr()
            root = BinaryOp(op, root, b)
        return root

    def multexpr(self):
        root = self.unaryexpr()
        while self.LA(1) in MULOP:
            op = self.LT(1)
            self.t += 1
            b = self.unaryexpr()
            root = BinaryOp(op, root, b)
        return root

    def unaryexpr(self):
        if self.LA(1) in UNARYOP:
            op = self.LT(1)
            self.t += 1
            e = self.unaryexpr()
            return UnaryOp(op, e)
        elif self.isatom() or self.isgroup():
            return self.postexpr()
        else:
            print(f"missing unary expr at: {self.LT(1)}")

    def postexpr(self):
        e = self.atom()
        if self.LA(1)==LPAR:
            return self.funccall(e)
        if self.LA(1)==LSQB:
            return self.index(e)
        return e

    def atom(self):
        if self.LA(1) == LPAR:
            return self.subexpr()
        elif self.LA(1) == LSQB:
            return self.listatom()
        elif self.isatom() or self.isgroup() or self.LA(1)==COLON:
            atom = self.LT(1)
            self.t += 1  # match name or number
            return Atom(atom)
        else:
            print("error")

    def funccall(self, f):
        self.match(LPAR)
        el = None
        if self.LA(1)!=RPAR:
            el = self.exprlist()
        self.match(RPAR)
        return Call(f, el)

    def index(self, e):
        self.match(LSQB)
        el = self.exprlist()
        self.match(RSQB)
        return Index(e, el)

    def exprlist(self):
        elist = []
        e = self.expression()
        elist.append(e)
        while self.LA(1)==COMMA:
            self.match(COMMA)
            e = self.expression()
            elist.append(e)
        return elist# if len(elist)>1 else elist[0]

    def subexpr(self):
        self.match(LPAR)
        e = self.expression()
        self.match(RPAR)
        return SubExpr(e)

    def listatom(self):
        self.match(LSQB)
        e = self.exprlist()
        self.match(RSQB)
        return ListLiteral(e)

    def isatom(self):
        return self.LA(1) in {NUMBER, NAME, STRING, COLON}
        # return idstart(self.LA(1)) or self.LA(1).isdigit() or self.LA(1)==':'

    def isgroup(self):
        return self.LA(1)==LPAR or self.LA(1)==LSQB

    def LA(self, i):
        return self.LT(i).type

    def LT(self, i):
        ahead = self.t + i - 1
        if ahead >= len(self.tokens):
            return ENDMARKER
        return self.tokens[ahead]

    def match(self, type):
        if self.LA(1)!=type:
            print(f"mismatch token {self.LT(1)}, looking for {token.tok_name[type]}")
        self.t += 1


def mytokenize(s):
    tokensO = tokenize(BytesIO(s.encode('utf-8')).readline)
    tokens = []
    for tok in tokensO:
        type, value, _, _, _ = tok
        if type in {NUMBER, STRING, NAME, OP, ENDMARKER}:
            tokens.append(Token(tok.exact_type,value))
        else:
            # print("ignoring", type, value)
            pass

    # Scan for "a.b.c.d" type patterns and combine into NAME
    tokens2 = []
    i = 0
    while i<len(tokens):
        if tokens[i].type==NAME:
            start = i
            i += 1
            while tokens[i].type==DOT:
                i += 2 # skip over ". name"
            tokens2.append(Token(NAME,''.join(str(t) for t in tokens[start:i])))
        else:
            tokens2.append(tokens[i])
            i += 1
    tokens = tokens2
    return tokens#+[Token(ENDMARKER,"<EOF>")]


class dbg:
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
            p = PyExprParser(code)
            t = p.parse()
            try:
                t.eval(exc_frame)
            except IncrEvalTrap as exc:
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

def _shape(v):
    if hasattr(v, "shape"):
        if isinstance(v.shape, torch.Size):
            return list(v.shape)
        return v.shape
    return None
