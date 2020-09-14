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
from tsensor.ast import IncrEvalTrap, set_matrix_below
from tsensor.parsing import PyExprParser
import sys
import numpy as np
import torch

def check(s,expected):
    frame = sys._getframe()
    caller = frame.f_back
    p = PyExprParser(s)
    t = p.parse()
    bad_subexpr = None
    try:
        t.eval(caller)
    except IncrEvalTrap as exc:
        bad_subexpr = str(exc.offending_expr)
    assert bad_subexpr==expected


def test_missing_var():
    a = 3
    c = 5
    check("a+b+c", "b")
    check("z+b+c", "z")

def test_matrix_mult():
    W = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[1,2,3]])
    check("W@b+torch.abs(b)", "W@b")

def test_bad_arg():
    check("torch.abs('foo')", "torch.abs('foo')")

def test_parens():
    a = 3
    b = 4
    c = 5
    check("(a+b)/0", "(a+b)/0")

def test_array_literal():
    a = torch.tensor([[1,2,3],[4,5,6]])
    b = torch.tensor([[1,2,3]])
    a+b
    check("a + b@2", """b@2""")

def test_array_literal2():
    a = torch.tensor([[1,2,3],[4,5,6]])
    b = torch.tensor([[1,2,3]])
    a+b
    check("(a+b)@2", """(a+b)@2""")
