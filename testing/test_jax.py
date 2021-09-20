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
import jax.numpy as jnp
import numpy as np
import tsensor


def test_dot():
    size = 5000
    x = np.random.normal(size=(size, size)).astype(np.float32)
    y = np.random.normal(size=(5, 1)).astype(np.float32)

    msg = ""
    try:
        with tsensor.clarify():
            z = jnp.dot(x, y).block_until_ready()
    except TypeError as e:
        msg = e.args[0]

    expected = (
        "Incompatible shapes for dot: got (5000, 5000) and (5, 1).\n"
        + "Cause: jnp.dot(x, y) tensor arg x w/shape (5000, 5000), arg y w/shape (5, 1)"
    )
    assert msg == expected


def test_scalar_arg():
    size = 5000
    x = np.random.normal(size=(size, size)).astype(np.float32)

    msg = ""
    try:
        with tsensor.clarify():
            z = jnp.dot(x, "foo")
    except TypeError as e:
        msg = e.args[0]

    expected = (
        "data type 'foo' not understood\n"
        + 'Cause: jnp.dot(x, "foo") tensor arg x w/shape (5000, 5000)'
    )
    assert msg == expected


def test_mmul():
    W = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([9, 10, 11])

    msg = ""
    try:
        with tsensor.clarify():
            y = W @ b
    except TypeError as e:
        msg = e.args[0]

    expected = (
        "dot_general requires contracting dimensions to have the same shape, got [2] and [3].\n"
        + "Cause: @ on tensor operand W w/shape (2, 2) and operand b w/shape (3,)"
    )
    assert msg == expected


def test_fft():
    "Test a library function that doesn't have a shape related message in the exception."
    x = np.exp(2j * np.pi * np.arange(8) / 8)
    msg = ""
    try:
        with tsensor.clarify():
            y = jnp.fft.fft(x, norm="something weird")
    except BaseException as e:
        msg = e.args[0]
        print(msg)

    expected = (
        "jax.numpy.fft.fft only supports norm=None, got something weird\n"
        + 'Cause: jnp.fft.fft(x, norm="something weird") tensor arg x w/shape (8,)'
    )
    assert msg == expected
