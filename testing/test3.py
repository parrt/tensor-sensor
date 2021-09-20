import tsensor
import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.constant([[1, 2], [3, 4]])
b = tf.reshape(tf.constant([[9, 10]]), (2, 1))
x = tf.reshape(tf.constant([[8, 5, 7]]), (3, 1))
z = 0

# tsensor.parse("z /= b + x * 3", hush_errors=False)

# with tsensor.clarify(show='viz'):
#     b + x * 3

fig, ax = plt.subplots(1, 1)
tsensor.pyviz("b + x", ax=ax)
plt.show()
# with tsensor.explain():
#     b + x
