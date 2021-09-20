import tsensor
import numpy as np


def f():
    # Currently can't handle double assign
    a = b = np.ones(1) @ np.ones(2)


def A():
    with tsensor.clarify():
        f()


def test_nested():
    msg = ""
    try:
        A()
    except BaseException as e:
        msg = e.args[0]

    expected = "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 1)"
    assert msg == expected
