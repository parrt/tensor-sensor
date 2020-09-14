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
import tsensor.ast
import sys
import numpy as np


def check(s,expected):
    frame = sys._getframe()
    caller = frame.f_back
    p = PyExprParser(s)
    t = p.parse()
    result = t.eval(caller)
    assert str(result)==str(expected)


def test_int():
    check("34", 34)

def test_assign():
    check("a = 34", 34)

def test_var():
    a = 34
    check("a", 34)

def test_member_var():
    class A:
        def __init__(self):
            self.a = 34
    x = A()
    check("x.a", 34)

def test_member_func():
    class A:
        def f(self, a):
            return a+4
    x = A()
    check("x.f(30)", 34)

def test_index():
    a = [1,2,3]
    check("a[2]", 3)

def test_add():
    a = 3
    b = 4
    c = 5
    check("a+b+c", 12)

def test_add_mul():
    a = 3
    b = 4
    c = 5
    check("a+b*c", 23)

def test_parens():
    a = 3
    b = 4
    c = 5
    check("(a+b)*c", 35)

def test_list_literal():
    a = [[1,2,3],[4,5,6]]
    check("a", """[[1, 2, 3], [4, 5, 6]]""")


def test_np_literal():
    a = np.array([[1,2,3],[4,5,6]])
    check("a*2", """[[ 2  4  6]\n [ 8 10 12]]""")


def test_np_add():
    a = np.array([[1,2,3],[4,5,6]])
    check("a+a", """[[ 2  4  6]\n [ 8 10 12]]""")


def test_np_add2():
    a = np.array([[1,2,3],[4,5,6]])
    check("a+a+a", """[[ 3  6  9]\n [12 15 18]]""")
