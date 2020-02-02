# coding: utf-8
# @Time    : 2020/1/26 9:30 下午
# @Author  : douwentao
import tensorflow as tf


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variables(shape, regularizer):
    weights = tf.get_variable(name="weights",
                              shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def inference(input_tesnor, regularizer):
    # 第一层
    with tf.variable_scope('layer1'):
        weights = get_weight_variables([INPUT_NODE, LAYER1_NODE], regularizer)
        bias = tf.get_variable(name="bias",
                               shape=[LAYER1_NODE],
                               initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tesnor, weights) + bias)

    # 第二层作为输出层
    with tf.variable_scope('layer2'):
        weights = get_weight_variables(
            [LAYER1_NODE, OUTPUT_NODE],
            regularizer
        )
        bias = tf.get_variable(name='bias',
                               shape=[OUTPUT_NODE],
                               initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1, weights) + bias
    return layer2