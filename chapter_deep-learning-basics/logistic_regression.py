# coding: utf-8
# @Time    : 2020/1/16 1:13 下午
# @Author  : douwentao
import numpy as np
import tensorflow as tf
from utils import obt_shape


def sigmoid(x):
    return tf.divide(1, 1 + tf.exp(-x))


def construct_loss(pred, y, method='log_loss'):
    assert obt_shape(pred)[0] == obt_shape(y)[0], "预测标签和真实标签不相同"
    sample_num = obt_shape(y)[0]
    if method == 'log_loss':
        front_part = tf.multiply(y, tf.log(pred))
        back_part = tf.multiply(tf.subtract(1.0, y), tf.log(tf.subtract(1.0, pred)))
        sum_loss = tf.reduce_sum(tf.add(front_part, back_part))
        loss = -1 / sample_num * sum_loss
    else:
        raise Exception("不支持损失函数: {f} 计算".format(f=method))

    return loss


def prepare_data(feature_dimension):

    x1 = np.linspace(0, 10, feature_dimension * 5) + 5
    x1 = np.reshape(x1, [5, feature_dimension])
    x2 = np.linspace(0, 10, feature_dimension * 5) - 5
    x2 = np.reshape(x2, [5, feature_dimension])
    x = np.vstack([x1, x2])
    y = np.vstack([np.ones((5, 1)), np.zeros((5, 1))])
    # 数据不多，可直接转换为张量
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    return x, y


def logistic_regression_test():

    feature_dimension = 2
    # 初始化数据, 10个样本,样本特征维度为feature_dimension
    x, y = prepare_data(feature_dimension)
    w = tf.Variable(tf.truncated_normal(shape=[1, feature_dimension], mean=0.0, stddev=0.2, seed=2020, dtype=tf.float32))
    b = tf.Variable(tf.truncated_normal(shape=[1], mean=0.0, stddev=0.2, seed=2020, dtype=tf.float32))
    # 定义超参数
    learning_rate = 0.001
    loop = 10000
    # 模型训练
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(loop):
        if i != 0 and i % 100 == 0:
            print(sess.run(cur_loss))
        pred = tf.matmul(x, tf.transpose(w))
        pred = tf.add(pred, b)
        pred = sigmoid(pred)
        cur_loss = construct_loss(pred, y)
        update_w = tf.reduce_sum(tf.multiply(tf.subtract(pred, y), x), axis=0)
        update_w = tf.expand_dims(update_w, 0)
        update_b = tf.reduce_sum(tf.subtract(pred, y), axis=0)
        w = w - tf.multiply(learning_rate, update_w)
        b = b - tf.multiply(learning_rate, update_b)
    sess.close()


if __name__ == '__main__':
    logistic_regression_test()