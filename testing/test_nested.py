# Test for https://github.com/parrt/tensor-sensor/issues/18
# Nested clarify's and all catch exception

import tsensor
import numpy as np


def f():
    np.ones(1) @ np.ones(2)


def A():
    with tsensor.clarify():
        f()


def B():
    with tsensor.clarify():
        A()


def test_nested():
    msg = ""
    try:
        B()
    except BaseException as e:
        msg = e.args[0]

    expected = (
        "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 1)\n"
        + "Cause: @ on tensor operand np.ones(1) w/shape (1,) and operand np.ones(2) w/shape (2,)"
    )
    assert msg == expected
