import tsensor
import tensorflow as tf

W = tf.constant([[1, 2], [3, 4]])
b = tf.reshape(tf.constant([[9, 10]]), (2, 1))
x = tf.reshape(tf.constant([[8, 5, 7]]), (3, 1))


def foo(): bar()

def bar(): b + x * 3

with tsensor.clarify():
    foo()
