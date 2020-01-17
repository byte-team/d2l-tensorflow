# coding: utf-8
# @Time    : 2020/1/17 2:39 下午
# @Author  : douwentao
import tensorflow as tf


class ReluActivator(object):
    @staticmethod
    def forward(x):
        return tf.maximum(0, x)

    @staticmethod
    def backward(y):
        return tf.cast(tf.greater(y, 0), dtype=tf.float32)


class SigmoidActivator(object):
    @staticmethod
    def forward(x):
        return tf.divide(1, 1 + tf.exp(-x))

    @staticmethod
    def backward(y):
        return tf.multiply(y, tf.subtract(1, y))


class TanhActivator(object):
    @staticmethod
    def forward(x):
        up_part = tf.subtract(tf.exp(x), tf.exp(-x))
        down_part = tf.add(tf.exp(x), tf.exp(-x))
        return tf.divide(up_part, down_part)

    @staticmethod
    def backward(y):
        return tf.subtract(1, tf.pow(y, 2))