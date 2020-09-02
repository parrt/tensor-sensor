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
import sys
from collections import namedtuple
from tokenize import tokenize, TokenInfo,\
    NUMBER, STRING, NAME, OP, ENDMARKER, LPAR, LSQB, RPAR, RSQB, EQUAL, COMMA, COLON,\
    PLUS, MINUS, STAR, SLASH, AT, PERCENT, TILDE, DOT
import token
from io import BytesIO

ADDOP     = {PLUS, MINUS}
MULOP     = {STAR, SLASH, AT, PERCENT}
UNARYOP   = {TILDE}
# OPERATORS = {'+', '-', '*', '/', '@', '%', '!', '~'}
# SYMBOLS   = OPERATORS.union({'(', ')', '[', ']', '=', ',', ':'})

# def idstart(c):
#     return c[0].isalpha() or c[0]=='_'
#
# def idchar(c): # include '.'; assume single char here
#     return c.isalpha() or c.isdigit() or c == '_' or c == '.'


# Parse tree definitions
# I found package ast in python3 lib after I built this. whoops. No biggie.
# This tree structure is easier to visit for my purposes here.

class ParseTreeNode:
    def eval(self, frame):
        "Evaluate the expression represented by this (sub)tree in context of frame"
        return eval(str(self), frame.f_locals, frame.f_globals)
    @property
    def left(self): return None
    @property
    def right(self): return None
    def __str__(self):
        pass
    def __repr__(self):
        args = [v+'='+self.__dict__[v].__repr__() for v in self.__dict__]
        args = ','.join(args)
        return f"{self.__class__.__name__}({args})"

class Assign(ParseTreeNode):
    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs
    def eval(self, frame):
        "Only consider rhs of assignment"
        return eval(str(self.rhs), frame.f_locals, frame.f_globals)
    @property
    def left(self): return self.lhs
    @property
    def right(self): return self.rhs
    def __str__(self):
        return str(self.lhs)+'='+str(self.rhs)

class Call(ParseTreeNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args
    @property
    def left(self): return self.args
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
    @property
    def left(self): return self.index
    def __str__(self):
        i = self.index
        if isinstance(i,list):
            i = ','.join(str(v) for v in i)
        return f"{self.name}[{i}]"

class BinaryOp(ParseTreeNode):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b
    @property
    def left(self): return self.a
    @property
    def right(self): return self.b
    def __str__(self):
        return f"{self.a}{self.op}{self.b}"

class UnaryOp(ParseTreeNode):
    def __init__(self, op, opnd):
        self.op = op
        self.opnd = opnd
    @property
    def left(self): return self.opnd
    def __str__(self):
        return f"{self.op}{self.opnd}"

class ListLiteral(ParseTreeNode):
    def __init__(self, elems):
        self.elems = elems
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
        self.e = e
    @property
    def left(self): return self.e
    def __str__(self):
        return f"({self.e})"

class Atom(ParseTreeNode):
    def __init__(self, nametok):
        self.nametok = nametok
    def __repr__(self):
        return self.nametok.value
    def __str__(self):
        return self.nametok.value

class String(ParseTreeNode):
    def __init__(self, stok):
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
        return elist if len(elist)>1 else elist[0]

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


class Token:
    def __init__(self, type, value):
        self.type, self.value = type, value
    def __repr__(self):
        return f"<{token.tok_name[self.type]}:{self.value}>"
    def __str__(self):
        return self.value

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

# def mytokenize(code):
#     n = len(code)
#     i = 0
#     tokens = []
#     while i<len(code):
#         if idstart(code[i]):
#             v = []
#             while i<n and idchar(code[i]):
#                 v.append(code[i])
#                 i += 1
#             tokens.append(''.join(v))
#         elif code[i].isdigit():
#             num = []
#             while i<n and code[i].isdigit():
#                 num.append(code[i])
#                 i += 1
#             tokens.append(''.join(num))
#         elif code[i] in SYMBOLS:
#             op = code[i]
#             i += 1
#             tokens.append(op)
#         elif code[i] in {'r','u','f'} and code[i+1] in {'\'', '"'}:
#             quote = code[i]
#             i += 1
#             start = i
#             while i<n and code[i]!=quote:
#                 if code[i]=='\\':
#                     i += 1
#                 i += 1
#             tokens.append(code[start:i])
#         elif code[i] in {'\'', '"'}:
#             quote = code[i]
#             i += 1
#             start = i
#             while i<n and code[i]!=quote:
#                 if code[i]=='\\':
#                     i += 1
#                 i += 1
#             tokens.append(code[start:i])
#         elif code[i] in {' ','\t'}:
#             i += 1
#         else:
#             print("skip", code[i])
#             i += 1
#     print("tokens", tokens)
#     return tokens + [EOF]


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

            augment = ""
            try:
                p = PyExprParser(code)
                t = p.parse()
                try:
                    incr_eval(t, exc_frame)
                except IncrEvalTrap as exc:
                    subexpr = exc.expr
                    # print("trapped at", subexpr)
                    if subexpr.left is not None:
                        left = subexpr.left.eval(exc_frame)
                        # print(subexpr.left, left)
                        if self.has_shape(left):
                            # print(subexpr.left, "shape", left.shape)
                            augment += f"{subexpr.left}.shape={left.shape}"
                    if subexpr.right is not None:
                        right = subexpr.right.eval(exc_frame)
                        # print(subexpr.right, right)
                        if self.has_shape(right):
                            # print(subexpr.right, "shape", right.shape)
                            augment += f"\n{subexpr.right}.shape={right.shape}"
            except BaseException as e:
                print(f"exception while eval({code})", e)

            # Reuse exception but overwrite the message
            exc_value.args = [exc_value.args[0]+"\n"+augment]

    def has_shape(self, v):
        return hasattr(v, "shape")

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


class IncrEvalTrap(BaseException):
    def __init__(self, expr):
        self.expr = expr # where in tree did we get exception?


def incr_eval(tree, frame):
    "Incrementally evaluate all subexpressions, looking for operation that fails; return that subtree"
    if tree is None:
        return
    if isinstance(tree, list): # must be args list or expr list
        for t in tree:
            incr_eval(t, frame)
        return
    if isinstance(tree, Assign):
        incr_eval(tree.right, frame)
    elif tree.left is not None and tree.right is not None: # binary
        incr_eval(tree.left, frame)
        incr_eval(tree.right, frame)
    elif tree.left is not None: # unary
        incr_eval(tree.left, frame)
    try:
        tree.eval(frame) # try to do this operator
    except:
        raise IncrEvalTrap(tree)
    # else all is well, just return to larger subexpr up the tree

