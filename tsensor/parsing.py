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
from tokenize import tokenize,\
    NUMBER, STRING, NAME, OP, ENDMARKER, LPAR, LSQB, RPAR, RSQB, EQUAL, COMMA, COLON,\
    PLUS, MINUS, STAR, SLASH, AT, PERCENT, TILDE, DOT

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
UNARYOP   = {TILDE}


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
    return tokens


class PyExprParser:
    def __init__(self, code):
        self.code = code
        self.tokens = mytokenize(code)
        self.t = 0 # current lookahead

    def parse(self):
        # print("\nparse", self.code)
        # print(self.tokens)
        # only process assignments and expressions
        if not keyword.iskeyword(self.tokens[0].value):
            s = self.statement()
            self.match(ENDMARKER)
            return s
        return None

    def statement(self):
        lhs = self.expression()
        if self.LA(1) == EQUAL:
            eq = self.LT(1)
            self.t += 1
            rhs = self.expression()
            return tsensor.ast.Assign(eq,lhs,rhs)
        return lhs

    def expression(self):
        return self.addexpr()

    def addexpr(self):
        root = self.multexpr()
        while self.LA(1) in ADDOP:
            op = self.LT(1)
            self.t += 1
            b = self.multexpr()
            root = tsensor.ast.BinaryOp(op, root, b)
        return root

    def multexpr(self):
        root = self.unaryexpr()
        while self.LA(1) in MULOP:
            op = self.LT(1)
            self.t += 1
            b = self.unaryexpr()
            root = tsensor.ast.BinaryOp(op, root, b)
        return root

    def unaryexpr(self):
        if self.LA(1) in UNARYOP:
            op = self.LT(1)
            self.t += 1
            e = self.unaryexpr()
            return tsensor.ast.UnaryOp(op, e)
        elif self.isatom() or self.isgroup():
            return self.postexpr()
        else:
            print(f"missing unary expr at: {self.LT(1)}")

    # def memberexpr(self):
    #     root = self.postexpr()
    #     while self.LA(1)==DOT:
    #         self.match(DOT)
    #         m = self.postexpr()
    #         root = Member(root, m)
    #     return root

    def postexpr(self):
        root = self.atom()
        while self.LA(1) in {LPAR, LSQB, DOT}:
            if self.LA(1)==LPAR:
                self.match(LPAR)
                el = []
                if self.LA(1) != RPAR:
                    el = self.exprlist()
                self.match(RPAR)
                root = tsensor.ast.Call(root, el)
            if self.LA(1)==LSQB:
                self.match(LSQB)
                el = self.exprlist()
                self.match(RSQB)
                root = tsensor.ast.Index(root, el)
            if self.LA(1)==DOT:
                op = self.match(DOT)
                m = self.match(NAME)
                m = tsensor.ast.Atom(m)
                root = tsensor.ast.Member(op, root, m)
        return root

    def atom(self):
        if self.LA(1) == LPAR:
            return self.subexpr()
        elif self.LA(1) == LSQB:
            return self.listatom()
        elif self.LA(1) in {NUMBER, NAME, STRING, COLON, COLON}:
            atom = self.LT(1)
            self.t += 1
            return tsensor.ast.Atom(atom)
        else:
            print("error")

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
        return tsensor.ast.SubExpr(e)

    def listatom(self):
        self.match(LSQB)
        e = self.exprlist()
        self.match(RSQB)
        return tsensor.ast.ListLiteral(e)

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
        tok = self.LT(1)
        self.t += 1
        return tok


def parse(statement:str):
    "Parse statement and return ast and token objects."
    p = tsensor.parsing.PyExprParser(statement)
    return p.parse(), p.tokens
