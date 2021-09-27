import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import torch

import tsensor as ts


def test_dtypes():
    values = [
        # Numpy
        (np.random.randint(1, 10, size=(10, 2, 5)), "int64"),
        (np.random.randint(1, 10, size=(10, 2, 5), dtype="int8"), "int8"),
        (np.random.normal(size=(5, 1)).astype(np.float32), "float32"),
        (np.random.normal(size=(5, 1)).astype(np.float32), "float32"),
        (np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]), "str320,int32,float32"),
        # Jax
        (jnp.array([[1, 2], [3, 4]]), "int32"),
        (jnp.array([[1, 2], [3, 4]], dtype="int8"), "int8"),
        # Tensorflow
        (tf.constant([[1, 2], [3, 4]]), "int32"),
        (tf.constant([[1, 2], [3, 4]], dtype="int64"), "int64"),
        # Pytorch
        (torch.tensor([[1, 2], [3, 4]]), "int64"),
        (torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), "int32"),
    ]

    for value, expected in values:
        assert ts.analysis._dtype(value) == expected
