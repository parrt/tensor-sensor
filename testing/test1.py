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
import numpy as np
import dis
import torch 

a = np.array([1,2,3])
g = globals()

from inspect import currentframe, getframeinfo, stack
def info():
    prev = stack()[1]
    return prev.filename, prev.lineno

def prevline():
    prev = stack()[1]
    filename, line = prev.filename, prev.lineno
    line -= 1 # get previous line
    with open(filename, "r") as f:
        code = f.read()
    lines = code.split('\n')
    if line==1:
        return None
    return lines[line-1] # indexed from 1


def nextline(n):
    prev = stack()[1]
    filename, line = prev.filename, prev.lineno
    line += n # get n lines ahead
    with open(filename, "r") as f:
        code = f.read()
    lines = code.split('\n')
    if line==1:
        return None
    return lines[line-1] # indexed from 1


def f():
    W = np.array([[1,2],[3,4]])
    b = np.array([9,10]).reshape(2,1)
    x = np.array([4,5]).reshape(2,1)
    code = nextline(2).strip()
    exec(code)
    b = W @ b + x
    loc = locals()
    print(eval(compile("b*x", "", "eval")))

class dbg:
    def __enter__(self):
        prev = stack()[1]
        filename, line = prev.filename, prev.lineno
        with open(filename, "r") as f:
            code = f.read()
        lines = code.split('\n')
        line += 1 # next line
        code = lines[line-1].strip() # index from 0
        print("code to dbg", code)
        # c = compile(code, "", "exec")
        c = dis.Bytecode(code)
        print(c.dis())
        VARLOADS = {'LOAD_NAME','LOAD_GLOBAL'}
        varrefs = [I.argval for I in c if I.opname in VARLOADS]
        funcrefs = [I.argval for I in c if I.opname in {'LOAD_METHOD'}]
        print("symbols",set(varrefs), set(funcrefs))
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")

def hi(): print("hi"); return 99


W = np.array([[1, 2], [3, 4]])
b = np.array([9, 10]).reshape(2, 1)
x = np.array([4, 5]).reshape(2, 1)

with dbg():
    z = torch.sigmoid(self.Whz @ h + self.Uxz @ x + self.bz)
    b = W @ b + np.abs(x)

# dis.dis("f()")
# dis.dis("a.f()")
# dis.dis("np.pi")
# dis.dis("a+3")