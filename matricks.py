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
import typing
from collections import namedtuple

ADDOP     = {'+', '-'}
MULOP     = {'*', '/', '@', '%'}
UNARYOP   = {'!', '~'}
OPERATORS = {'+', '-', '*', '/', '@', '%', '!', '~'}
SYMBOLS   = OPERATORS.union({'(', ')', '[', ']', '=', ',', ':'})
EOF       = '<EOF>'

def idstart(c):
    return c[0].isalpha() or c[0]=='_'

def idchar(c): # include '.'; assume single char here
    return c.isalpha() or c.isdigit() or c == '_' or c == '.'


# Parse tree definitions

class ParseTreeNode:
    def __str__(self):
        return "<ParseTreeNode>"
    def __repr__(self):
        args = [v+'='+self.__dict__[v].__repr__() for v in self.__dict__]
        args = ','.join(args)
        return f"{self.__class__.__name__}({args})"

class Assign(ParseTreeNode):
    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs
    def __str__(self):
        return str(self.lhs)+'='+str(self.rhs)

class Call(ParseTreeNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args
    def __str__(self):
        if isinstance(self.args,list):
            args_ = ','.join([str(a) for a in self.args])
        else:
            args_ = str(self.args)
        return f"{self.name}({args_})"

class Index(ParseTreeNode):
    def __init__(self, name, index):
        self.name = name
        self.index = index
    def __str__(self):
        i = self.index
        if isinstance(i,list):
            i = ','.join(str(v) for v in i)
        return f"{self.name}[{i}]"

class BinaryOp(ParseTreeNode):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b
    def __str__(self):
        return f"{self.a}{self.op}{self.b}"

class UnaryOp(ParseTreeNode):
    def __init__(self, op, opnd):
        self.op = op
        self.opnd = opnd
    def __str__(self):
        return f"{self.op}{self.opnd}"

class ListLiteral(ParseTreeNode):
    def __init__(self, elems):
        self.elems = elems
    def __str__(self):
        return f"[{','.join(str(e) for e in self.elems)}]"

class SubExpr(ParseTreeNode):
    # record parens for later display to keep precedence
    def __init__(self, e):
        self.e = e
    def __str__(self):
        return f"({self.e})"

class Atom(ParseTreeNode):
    def __init__(self, s):
        self.s = s
    def __repr__(self):
        return self.s
    def __str__(self):
        return self.s


class PyExprParser:
    def __init__(self, code):
        self.code = code
        self.tokens = tokenize(code)
        self.t = 0 # current lookahead

    def parse(self):
        print("\nparse", self.code)
        print(self.tokens)
        s = self.statement()
        self.match(EOF)
        return s

    def statement(self):
        lhs = self.expression()
        rhs = None
        if self.LA(1) == '=':
            self.t += 1
            rhs = self.expression()
            return Assign(lhs,rhs)
        return lhs

    def expression(self):
        return self.addexpr()

    def addexpr(self):
        elist = []
        root = self.multexpr()
        while self.LA(1) in ADDOP:
            op = self.LA(1)
            elist.append(self.LA(1))
            self.t += 1
            b = self.multexpr()
            root = BinaryOp(op, root, b)
        return root

    def multexpr(self):
        elist = []
        root = self.unaryexpr()
        while self.LA(1) in MULOP:
            op = self.LA(1)
            elist.append(self.LA(1))
            self.t += 1
            b = self.unaryexpr()
            root = BinaryOp(op, root, b)
        return root

    def unaryexpr(self):
        if self.LA(1) in UNARYOP:
            op = self.LA(1)
            self.t += 1
            e = self.unaryexpr()
            return UnaryOp(op, e)
        elif self.isatom() or self.isgroup():
            return self.postexpr()
        else:
            print(f"missing unary expr at: {self.LA(1)}")

    def postexpr(self):
        e = self.atom()
        if self.LA(1)=='(':
            return self.funccall(e)
        if self.LA(1) == '[':
            return self.index(e)
        return e

    def atom(self):
        if self.LA(1) == '(':
            return self.subexpr()
        elif self.LA(1) == '[':
            return self.listatom()
        elif self.isatom() or self.isgroup():
            atom = self.LA(1)
            self.t += 1  # match name or number
            return Atom(atom)
        else:
            print("error")

    def funccall(self, f):
        self.match('(')
        el = None
        if self.LA(1)!=')':
            el = self.exprlist()
        self.match(')')
        return Call(f, el)

    def index(self, e):
        self.match('[')
        el = self.exprlist()
        self.match(']')
        return Index(e, el)

    def exprlist(self):
        elist = []
        e = self.expression()
        elist.append(e)
        while self.LA(1)==',':
            self.match(',')
            e = self.expression()
            elist.append(e)
        return elist if len(elist)>1 else elist[0]

    def subexpr(self):
        self.match('(')
        e = self.expression()
        self.match(')')
        return SubExpr(e)

    def listatom(self):
        self.match('[')
        e = self.exprlist()
        self.match(']')
        return ListLiteral(e)

    def isatom(self):
        return idstart(self.LA(1)) or self.LA(1).isdigit() or self.LA(1)==':'

    def isgroup(self):
        return self.LA(1)=='(' or self.LA(1)=='['

    def LA(self, i):
        ahead = self.t + i - 1
        if ahead >= len(self.tokens):
            return EOF
        return self.tokens[ahead]

    def match(self, token):
        if self.LA(1)!=token:
            print(f"mismatch token {self.LA(1)}, looking for {token}")
        self.t += 1


def tokenize(code):
    n = len(code)
    i = 0
    tokens = []
    while i<len(code):
        if idstart(code[i]):
            v = []
            while i<n and idchar(code[i]):
                v.append(code[i])
                i += 1
            tokens.append(''.join(v))
        elif code[i].isdigit():
            num = []
            while i<n and code[i].isdigit():
                num.append(code[i])
                i += 1
            tokens.append(''.join(num))
        elif code[i] in SYMBOLS:
            op = code[i]
            i += 1
            tokens.append(op)
        elif code[i] in {' ','\t'}:
            i += 1
        else:
            print("skip", code[i])
            i += 1
    return tokens + [EOF]


class dbg:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            if not self.is_interesting_exception(exc_value):
                return
            print("exception:", exc_value, exc_traceback)
            # traceback.print_tb(exc_traceback, limit=5, file=sys.stdout)
            exc_frame = self.deepest_frame(exc_traceback)
            module, name, filename, line, code = self.info(exc_frame)
            print('info', module, name, filename, line, code)
            #raise RuntimeError("foo") from exc_value
            # Reuse exception but overwrite the message
            exc_value.args = ["was:" + exc_value.args[0]]

    def is_interesting_exception(self, e):
        sentinels = {'matmul', 'THTensorMath', 'tensor', 'tensors', 'dimension'}
        msg = e.args[0]
        return sum([s in msg for s in sentinels])>0

    def deepest_frame(self, exc_traceback):
        tb = exc_traceback
        while tb.tb_next != None:
            tb = tb.tb_next
        return tb.tb_frame

    def info(self, frame):
        module = frame.f_globals['__name__']
        info = inspect.getframeinfo(frame)
        code = info.code_context[0].strip()
        filename, line = info.filename, info.lineno
        name = info.function
        return module, name, filename, line, code

