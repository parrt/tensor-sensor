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
from tokenize import tokenize, \
    NUMBER, STRING, NAME, OP, ENDMARKER, LPAR, LSQB, RPAR, RSQB, COMMA, COLON,\
    PLUS, MINUS, STAR, SLASH, AT, PERCENT, TILDE, DOT,\
    NOTEQUAL, PERCENTEQUAL, AMPEREQUAL, DOUBLESTAREQUAL, STAREQUAL, PLUSEQUAL,\
    MINEQUAL, DOUBLESLASHEQUAL, SLASHEQUAL, LEFTSHIFTEQUAL,\
    LESSEQUAL, EQUAL, EQEQUAL, GREATEREQUAL, RIGHTSHIFTEQUAL, ATEQUAL,\
    CIRCUMFLEXEQUAL, VBAREQUAL

import tsensor.ast


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
    """My own version of a token, with content copied from Python's TokenInfo object."""
    def __init__(self, type, value,
                 index,      # token index
                 cstart_idx, # char start
                 cstop_idx,  # one past char end index so text[start_idx:stop_idx] works
                 line):
        self.type, self.value, self.index, self.cstart_idx, self.cstop_idx, self.line = \
            type, value, index, cstart_idx, cstop_idx, line
    def __repr__(self):
        return f"<{token.tok_name[self.type]}:{self.value},{self.cstart_idx}:{self.cstop_idx}>"
    def __str__(self):
        return self.value


def mytokenize(s):
    "Use Python's tokenizer to lex s and collect my own token objects"
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
    """
    A recursive-descent parser for subset of Python expressions and assignments.
    There is a built-in parser, but I only want to process Python code  this library
    can handle and I also want my own kind of abstract syntax tree. Constantly,
    it's easier if I just parse the code I care about and ignore everything else.
    Building this parser was certainly no great burden.
    """
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
        if self.tokens[0].value=='return' or not keyword.iskeyword(self.tokens[0].value):
            if self.hush_errors:
                try:
                    root = self.assignment_or_return_or_expr()
                    self.match(ENDMARKER)
                except SyntaxError as e:
                    root = None
            else:
                root = self.assignment_or_return_or_expr()
                self.match(ENDMARKER)
        return root

    def assignment_or_return_or_expr(self):
        start = self.LT(1)
        if self.LA(1)==NAME and self.LT(1).value=='return':
            self.match(NAME)
            r = self.exprlist()
            stop = self.LT(-1)
            return tsensor.ast.Return(r,start,stop)
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
                    el = self.arglist()
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
        while self.LA(1)==COMMA and self.LA(2)!=RPAR: # could be trailing comma in a tuple like (3,4,)
            self.match(COMMA)
            e = self.expression()
            elist.append(e)
        return elist

    def arglist(self):
        elist = []
        if self.LA(1)==NAME and self.LA(2)==EQUAL:
            e = self.arg()
        else:
            e = self.expression()
        elist.append(e)
        while self.LA(1)==COMMA:
            self.match(COMMA)
            if self.LA(1) == NAME and self.LA(2)==EQUAL:
                e = self.arg()
            else:
                e = self.expression()
            elist.append(e)
        return elist

    def arg(self):
        start = self.LT(1)
        kwarg = self.match(NAME)
        eq = self.match(EQUAL)
        e = self.expression()
        kwarg = tsensor.ast.Atom(kwarg)
        stop = self.LT(-1)
        return tsensor.ast.Assign(eq, kwarg, e, start, stop)

    def subexpr(self):
        start = self.match(LPAR)
        e = self.exprlist()  # could be a tuple like (3,4) or even (3,4,)
        istuple = len(e)>1
        if self.LA(1)==COMMA:
            self.match(COMMA)
            istuple = True
        stop = self.match(RPAR)
        if istuple:
            return tsensor.ast.TupleLiteral(e, start, stop)
        subexpr = e[0]
        return tsensor.ast.SubExpr(subexpr, start, stop)

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
    """
    Parse statement and return ast and token objects.  Parsing errors from invalid code
    or code that I cannot parse are ignored unless hush_hush_errors is False.
    """
    p = tsensor.parsing.PyExprParser(statement, hush_errors=hush_errors)
    return p.parse(), p.tokens
