# coding: utf-8
# @Time    : 2020/1/15 7:31 下午
# @Author  : douwentao
import numpy as np
import tensorflow as tf


def obt_shape(x):
    # 组合静态shape和动态shape
    shape_list = []
    static_shape_list = x.get_shape().as_list()
    none_index_list = []
    for i, val in enumerate(static_shape_list):
        if val == None:
            none_index_list.append(i)
    if len(none_index_list) == 0:
        return static_shape_list
    dynamic_shape_list = tf.shape(x)
    for i in range(len(static_shape_list)):
        if i in none_index_list:
            shape_list.append(dynamic_shape_list[i])
        else:
            shape_list.append(static_shape_list[i])
    return shape_list


def construct_loss(pred, y, method='l2'):
    # 计算损失
    assert obt_shape(pred)[0] == obt_shape(y)[0], "预测标签和真实标签不相同"
    sample_num = obt_shape(y)[0]
    if method == 'l2':
        # 损失计算过程，点操作
        res = tf.subtract(pred, y)
        res = tf.pow(res, 2)
        res = tf.reduce_sum(res)
        res = tf.sqrt(res)
        res = tf.divide(res, sample_num)
    else:
        raise Exception("不支持损失函数: {f} 计算".format(f=method))
    return res


def prepare_data(feature_dimension):
    x = np.linspace(0, 10, 10 * feature_dimension) + np.random.uniform(-1.5, 1.5, 10 * feature_dimension)
    y = np.linspace(0, 10, 10 * 1) + np.random.uniform(-1.5, 1.5, 10 * 1)
    x = np.reshape(x, [-1, 2])
    y = np.reshape(y, [-1, 1])
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def linear_regression():
    """
    梯度下降法
    :return:
    """
    feature_dimension = 2
    # 初始化数据
    x, y = prepare_data(feature_dimension)
    w = tf.Variable(tf.truncated_normal(shape=[1, feature_dimension], mean=0.0, stddev=0.2, seed=2020, dtype=tf.float32))
    b = tf.Variable(tf.truncated_normal(shape=[1], mean=0.0, stddev=0.2, seed=2020, dtype=tf.float32))
    # 超参数
    learning_rate = 0.001
    loop = 10000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    for i in range(loop):
        if i != 0 and i % 100 == 0:
            print(sess.run(cur_loss))
        # 获取当前预值（即前向传播）
        pred = tf.matmul(x, tf.transpose(w))
        # 广播机制
        pred = tf.add(pred, b)
        cur_loss = construct_loss(pred, y)
        # 获取当前步骤损失
        update_w = tf.reduce_sum(tf.multiply(tf.subtract(pred, y), x), axis=0)
        update_b = tf.reduce_sum(tf.subtract(pred, y), axis=0)
        # 反向传播
        w = w - tf.multiply(learning_rate, update_w)
        b = b - tf.multiply(learning_rate, update_b)
    sess.close()


def linear_regression_ls():
    """
    最小二乘法
    :return:
    """
    feature_dimension = 2
    # 初始化数据
    x, y = prepare_data(feature_dimension)
    tmp = tf.matmul(tf.transpose(x), x)
    front_part = tf.pow(tf.matmul(tf.transpose(x), x), -1)

    back_part = tf.matmul(tf.transpose(x), y)
    w = tf.matmul(front_part, back_part)
    pred = tf.matmul(x, w)
    cur_loss = construct_loss(pred, y)
    with tf.Session() as sess:
        print(sess.run(cur_loss))

if __name__ == '__main__':
    linear_regression()
    # linear_regression_ls()
    # feature_dimension = 2
    # # 初始化数据
    # x, y = prepare_data(feature_dimension)
    # tmp = tf.matmul(tf.transpose(x), x)
    # front_part = tf.pow(tmp, -1)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(x))
    #     print(sess.run(tf.matmul(tmp, front_part)))