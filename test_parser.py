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
from matricks import PyExprParser
import re

def check(s,expected):
    p = PyExprParser(s)
    t = p.parse()

    s = re.sub(r"\s+", "", s)
    result_str = str(t)
    result_str = re.sub(r"\s+", "", result_str)
    assert result_str==s

    result_repr = repr(t)
    result_repr = re.sub(r"\s+", "", result_repr)
    expected = re.sub(r"\s+", "", expected)
    print("result", result_repr)
    print("expected", expected)
    assert result_repr==expected


def test_index():
    check("a[:,i,j]", "Index(name=a, index=[:, i, j])")


def test_literal_list():
    check("[[1, 2], [3, 4]]",
          "ListLiteral(elems=[ListLiteral(elems=[1, 2]), ListLiteral(elems=[3, 4])])")


def test_literal_array():
    check("np.array([[1, 2], [3, 4]])",
          "Call(name=np.array, args=ListLiteral(elems=[ListLiteral(elems=[1, 2]), ListLiteral(elems=[3, 4])]))")


def test_method():
    check("h = torch.tanh(h)",
          "Assign(lhs=h, rhs=Call(name=torch.tanh, args=h))")


def test_parens():
    check("(a+b)*c", "BinaryOp(op=<STAR:*>,a=SubExpr(e=BinaryOp(op=<PLUS:+>,a=a,b=b)),b=c)")


def test_arith():
    check("(1-z)*h + z*h_",
          """BinaryOp(op=<PLUS:+>,
                      a=BinaryOp(op=<STAR:*>,
                                 a=SubExpr(e=BinaryOp(op=<MINUS:->,
                                                      a=1,
                                                      b=z)),
                                 b=h),
                      b=BinaryOp(op=<STAR:*>,a=z,b=h_))""")


def test_chained_op():
    check("a + b + c",
          """BinaryOp(op=<PLUS:+>,
                      a=BinaryOp(op=<PLUS:+>, a=a, b=b),
                      b=c)""")


def test_matrix_arith():
    check("self.Whz@h + self.Uxz@x + self.bz",
          """BinaryOp(op=<PLUS:+>,
                      a=BinaryOp(op=<PLUS:+>,
                                 a=BinaryOp(op=<AT:@>, a=self.Whz, b=h),
                                 b=BinaryOp(op=<AT:@>, a=self.Uxz, b=x)),
                      b=self.bz)""")


