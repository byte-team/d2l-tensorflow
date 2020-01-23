# coding: utf-8
# @Time    : 2020/1/23 12:06 下午
# @Author  : douwentao
import tensorflow as tf


def conv(x, filter_height, filter_width, num_filters, stride_y,
         stride_x, name, padding='SAME'):
    """
    普通卷积操作
    :param x: 输入
    :param filter_height: 过滤器高
    :param filter_width: 过滤器宽
    :param num_filters: 过滤器个数
    :param stride_y: y方向步长
    :param stride_x: x方向步长
    :param name: 命名
    :param padding:
    :return: 经过激活后的特征图
    """
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i,
                                         k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", shape=[filter_height, filter_width, input_channels, num_filters])
        bias = tf.get_variable("bias", shape=[num_filters])

        conv = convolve(x, weights)
        conv_result = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape().as_list())
        relu = tf.nn.relu(conv_result, name=scope.name)
        return relu



def fully_connected(x, in_num, out_num, name, relu=True):
    """
    全连接层
    :param x: 输入
    :param in_num: 输入特征维度
    :param out_num: 输入特征维度
    :param name: 命名
    :param relu: 是否使用relu激活函数
    :return:
    """
    with tf.variable_scope(name) as scope:
        # get_variable方法在获取命名空间中变量名所对应的变量，如果不存在，则新建变量
        weights = tf.get_variable("weights", shape=[in_num, out_num], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[out_num], dtype=tf.float32)
        act = tf.nn.xw_plus_b(x, weights, bias, name=scope.name)
        # 返回是否使用激活函数的结果
        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    """
    最大池化
    :param x: 输入，一般为四维[batch_size, height, width, depth]
    :param filter_height: 过滤器高
    :param filter_width: 过滤器宽
    :param stride_x: x方向过滤器步长
    :param stride_y: y方向过滤器步长
    :param name: 命名
    :param padding: 填充方式, 'valid'为不填充; 'SAME'为过滤后填充为与输入相同的维度
    :return:
    """
    return tf.nn.max_pool(x,
                          ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding,
                          name=name)

# BN优点
# 1. 防止过拟合（一般在使用sigmoid激活函数的时候，随着深度增大，激活函数的输入值会变大，导致导数变小）
# BN缺点
# 1. BN 依赖batch_size, 当batch_size小的时候, BN效果不佳；当batch_size大的时候，显存占用多;
# 2. BN 不适用于序列化数据的网络
# 3. BN只在训练的时候使用，inference的时候不使用


def dropout(x,keep_prob):
    return tf.nn.dropout(x, keep_prob)