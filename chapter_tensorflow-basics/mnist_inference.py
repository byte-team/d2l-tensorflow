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


def variable_summaries(var, name):
    # 记录变量信息
    with tf.name_scope('summaries'):
        # 通过tf.summary.histogram函数记录张量中元素的取值分布
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var, mean))))
        tf.summary.scalar('stddev/' + name, stddev)
    return


def nn_layer(input_tensor, input_dim, output_dim, layer_name, regularizer, activation=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(shape=[input_dim, output_dim],
                                                      stddev=0.1,
                                                      dtype=tf.float32))
            if regularizer:
                tf.add_to_collection('losses', regularizer(weights))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.constant(0.0, shape=[output_dim], dtype=tf.float32))
            variable_summaries(bias, layer_name + '/bias')
        with tf.name_scope('wx_plus_b'):
            preact = tf.matmul(input_tensor, weights) + bias
            tf.summary.histogram(layer_name + '/pre_activations', preact)
            act = activation(preact)
            tf.summary.histogram(layer_name + '/activations', act)
    return act


def inference_with_tensorboard(input_tensor, regularizer):
    layer1 = nn_layer(input_tensor,
                      INPUT_NODE,
                      LAYER1_NODE,
                      'layer1',
                      regularizer)

    layer2 = nn_layer(layer1,
                      LAYER1_NODE,
                      OUTPUT_NODE,
                      'layer2',
                      regularizer,
                      tf.identity)
    return layer2


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
