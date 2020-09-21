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
from io import BytesIO
import token
import keyword
from tokenize import tokenize, TokenInfo,\
    NUMBER, STRING, NAME, OP, ENDMARKER, LPAR, LSQB, RPAR, RSQB, COMMA, COLON,\
    PLUS, MINUS, STAR, SLASH, AT, PERCENT, TILDE, DOT,\
    NOTEQUAL, PERCENTEQUAL, AMPEREQUAL, DOUBLESTAREQUAL, STAREQUAL, PLUSEQUAL,\
    MINEQUAL, DOUBLESLASHEQUAL, SLASHEQUAL, COLONEQUAL, LEFTSHIFTEQUAL,\
    LESSEQUAL, EQUAL, EQEQUAL, GREATEREQUAL, RIGHTSHIFTEQUAL, ATEQUAL,\
    CIRCUMFLEXEQUAL, VBAREQUAL

import tsensor.ast

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
ASSIGNOP  = {NOTEQUAL,
             PERCENTEQUAL,
             AMPEREQUAL,
             DOUBLESTAREQUAL,
             STAREQUAL,
             PLUSEQUAL,
             MINEQUAL,
             DOUBLESLASHEQUAL,
             SLASHEQUAL,
             LEFTSHIFTEQUAL,
             LESSEQUAL,
             EQUAL,
             EQEQUAL,
             GREATEREQUAL,
             RIGHTSHIFTEQUAL,
             ATEQUAL,
             CIRCUMFLEXEQUAL,
             VBAREQUAL}
UNARYOP   = {TILDE}

class Token:
    def __init__(self, type, value,
                 index,  # token index
                 cstart_idx,  # char start
                 cstop_idx,  # one past char end index so text[start_idx:stop_idx] works
                 line):
        self.type, self.value, self.index, self.cstart_idx, self.cstop_idx, self.line = \
            type, value, index, cstart_idx, cstop_idx, line
    def __repr__(self):
        return f"<{token.tok_name[self.type]}:{self.value},{self.cstart_idx}:{self.cstop_idx}>"
    def __str__(self):
        return self.value


def mytokenize(s):
    tokensO = tokenize(BytesIO(s.encode('utf-8')).readline)
    tokens = []
    i = 0
    for tok in tokensO:
        type, value, start, end, _ = tok
        line = start[0]
        start_idx = start[1]
        stop_idx = end[1] # one past end index
        if type in {NUMBER, STRING, NAME, OP, ENDMARKER}:
            tokens.append(Token(tok.exact_type,value,i,start_idx,stop_idx,line))
            i += 1
        else:
            # print("ignoring", type, value)
            pass
    # It leaves ENDMARKER on end. set text to "<EOF>"
    tokens[-1].value = "<EOF>"
    # print(tokens)
    return tokens


class PyExprParser:
    def __init__(self, code:str, hush_errors=True):
        self.code = code
        self.hush_errors = hush_errors
        self.tokens = mytokenize(code)
        self.t = 0 # current lookahead

    def parse(self):
        # print("\nparse", self.code)
        # print(self.tokens)
        # only process assignments and expressions
        root = None
        if not keyword.iskeyword(self.tokens[0].value):
            if self.hush_errors:
                try:
                    root = self.assignment_or_expr()
                    self.match(ENDMARKER)
                except SyntaxError as e:
                    root = None
            else:
                root = self.assignment_or_expr()
                self.match(ENDMARKER)
        return root

    def assignment_or_expr(self):
        start = self.LT(1)
        lhs = self.expression()
        if self.LA(1) in ASSIGNOP:
            eq = self.LT(1)
            self.t += 1
            rhs = self.expression()
            stop = self.LT(-1)
            return tsensor.ast.Assign(eq,lhs,rhs,start,stop)
        return lhs

    def expression(self):
        return self.addexpr()

    def addexpr(self):
        start = self.LT(1)
        root = self.multexpr()
        while self.LA(1) in ADDOP:
            op = self.LT(1)
            self.t += 1
            b = self.multexpr()
            stop = self.LT(-1)
            root = tsensor.ast.BinaryOp(op, root, b, start, stop)
        return root

    def multexpr(self):
        start = self.LT(1)
        root = self.unaryexpr()
        while self.LA(1) in MULOP:
            op = self.LT(1)
            self.t += 1
            b = self.unaryexpr()
            stop = self.LT(-1)
            root = tsensor.ast.BinaryOp(op, root, b, start, stop)
        return root

    def unaryexpr(self):
        start = self.LT(1)
        if self.LA(1) in UNARYOP:
            op = self.LT(1)
            self.t += 1
            e = self.unaryexpr()
            stop = self.LT(-1)
            return tsensor.ast.UnaryOp(op, e, start, stop)
        elif self.isatom() or self.isgroup():
            return self.postexpr()
        else:
            self.error(f"missing unary expr at: {self.LT(1)}")

    def postexpr(self):
        start = self.LT(1)
        root = self.atom()
        while self.LA(1) in {LPAR, LSQB, DOT}:
            if self.LA(1)==LPAR:
                lp = self.LT(1)
                self.match(LPAR)
                el = []
                if self.LA(1) != RPAR:
                    el = self.exprlist()
                self.match(RPAR)
                stop = self.LT(-1)
                root = tsensor.ast.Call(root, lp, el, start, stop)
            if self.LA(1)==LSQB:
                lb = self.LT(1)
                self.match(LSQB)
                el = self.exprlist()
                self.match(RSQB)
                stop = self.LT(-1)
                root = tsensor.ast.Index(root, lb, el, start, stop)
            if self.LA(1)==DOT:
                op = self.match(DOT)
                m = self.match(NAME)
                m = tsensor.ast.Atom(m)
                stop = self.LT(-1)
                root = tsensor.ast.Member(op, root, m, start, stop)
        return root

    def atom(self):
        if self.LA(1) == LPAR:
            return self.subexpr()
        elif self.LA(1) == LSQB:
            return self.listatom()
        elif self.LA(1) in {NUMBER, NAME, STRING, COLON}:
            atom = self.LT(1)
            self.t += 1
            return tsensor.ast.Atom(atom)
        else:
            self.error("unknown or missing atom:"+str(self.LT(1)))

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
        start = self.LT(1)
        self.match(LPAR)
        e = self.expression()
        self.match(RPAR)
        stop = self.LT(-1)
        return tsensor.ast.SubExpr(e, start, stop)

    def listatom(self):
        start = self.LT(1)
        self.match(LSQB)
        e = self.exprlist()
        self.match(RSQB)
        stop = self.LT(-1)
        return tsensor.ast.ListLiteral(e, start, stop)

    def isatom(self):
        return self.LA(1) in {NUMBER, NAME, STRING, COLON}
        # return idstart(self.LA(1)) or self.LA(1).isdigit() or self.LA(1)==':'

    def isgroup(self):
        return self.LA(1)==LPAR or self.LA(1)==LSQB

    def LA(self, i):
        return self.LT(i).type

    def LT(self, i):
        if i==0:
            return None
        if i<0:
            return self.tokens[self.t + i] # -1 should give prev token
        ahead = self.t + i - 1
        if ahead >= len(self.tokens):
            return self.tokens[-1] # return last (end marker)
        return self.tokens[ahead]

    def match(self, type):
        if self.LA(1)!=type:
            self.error(f"mismatch token {self.LT(1)}, looking for {token.tok_name[type]}")
        tok = self.LT(1)
        self.t += 1
        return tok

    def error(self, msg):
        raise SyntaxError(msg)


def parse(statement:str, hush_errors=True):
    "Parse statement and return ast and token objects."
    p = tsensor.parsing.PyExprParser(statement, hush_errors=hush_errors)
    return p.parse(), p.tokens
