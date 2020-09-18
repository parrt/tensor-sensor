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
from tsensor.parsing import *
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
    # print("result", result_repr)
    # print("expected", expected)
    assert result_repr==expected


def test_assign():
    check("a = 3", "Assign(op=<EQUAL:=,2:3>,lhs=a,rhs=3)")


def test_index():
    check("a[:,i,j]", "Index(arr=a, index=[:, i, j])")


def test_index2():
    check("z = a[:]", "Assign(op=<EQUAL:=,2:3>,lhs=z,rhs=Index(arr=a,index=[:]))")

def test_index3():
    check("g.W[:,:,1]", "Index(arr=Member(op=<DOT:.,1:2>,obj=g,member=W),index=[:,:,1])")

def test_literal_list():
    check("[[1, 2], [3, 4]]",
          "ListLiteral(elems=[ListLiteral(elems=[1, 2]), ListLiteral(elems=[3, 4])])")


def test_literal_array():
    check("np.array([[1, 2], [3, 4]])",
          """
          Call(func=Member(op=<DOT:.,2:3>,obj=np,member=array),
               args=[ListLiteral(elems=[ListLiteral(elems=[1,2]),ListLiteral(elems=[3,4])])])
          """)


def test_method():
    check("h = torch.tanh(h)",
          "Assign(op=<EQUAL:=,2:3>,lhs=h,rhs=Call(func=Member(op=<DOT:.,9:10>,obj=torch,member=tanh),args=[h]))")


def test_field():
    check("a.b", "Member(op=<DOT:.,1:2>,obj=a,member=b)")


def test_member_func():
    check("a.f()", "Call(func=Member(op=<DOT:.,1:2>,obj=a,member=f),args=[])")


def test_field2():
    check("a.b.c", "Member(op=<DOT:.,3:4>,obj=Member(op=<DOT:.,1:2>,obj=a,member=b),member=c)")


def test_field_and_func():
    check("a.f().c", "Member(op=<DOT:.,5:6>,obj=Call(func=Member(op=<DOT:.,1:2>,obj=a,member=f),args=[]),member=c)")


def test_parens():
    check("(a+b)*c", "BinaryOp(op=<STAR:*,5:6>,lhs=SubExpr(e=BinaryOp(op=<PLUS:+,2:3>,lhs=a,rhs=b)),rhs=c)")


def test_field_array():
    check("a.b[34]", "Index(arr=Member(op=<DOT:.,1:2>,obj=a,member=b),index=[34])")


def test_field_array_func():
    check("a.b[34].f()", "Call(func=Member(op=<DOT:.,7:8>,obj=Index(arr=Member(op=<DOT:.,1:2>,obj=a,member=b),index=[34]),member=f),args=[])")


def test_arith():
    check("(1-z)*h + z*h_",
          """BinaryOp(op=<PLUS:+,8:9>,
                      lhs=BinaryOp(op=<STAR:*,5:6>,
                                 lhs=SubExpr(e=BinaryOp(op=<MINUS:-,2:3>,
                                                      lhs=1,
                                                      rhs=z)),
                                 rhs=h),
                      rhs=BinaryOp(op=<STAR:*,11:12>,lhs=z,rhs=h_))""")


def test_chained_op():
    check("a + b + c",
          """BinaryOp(op=<PLUS:+,6:7>,
                      lhs=BinaryOp(op=<PLUS:+,2:3>, lhs=a, rhs=b),
                      rhs=c)""")


def test_matrix_arith():
    check("self.Whz@h + Uxz@x + bz",
          """
          BinaryOp(op=<PLUS:+,19:20>,
                   lhs=BinaryOp(op=<PLUS:+,11:12>,
                                lhs=BinaryOp(op=<AT:@,8:9>,lhs=Member(op=<DOT:.,4:5>,obj=self,member=Whz),rhs=h),
                                rhs=BinaryOp(op=<AT:@,16:17>,lhs=Uxz,rhs=x)),
                   rhs=bz)
          """)

