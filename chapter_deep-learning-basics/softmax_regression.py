# coding: utf-8
# @Time    : 2020/1/16 3:53 下午
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


def prepare_data(feature_dimension):
    x1 = np.linspace(0, 10, feature_dimension * 5) + 5
    x1 = np.reshape(x1, [5, feature_dimension])
    x2 = np.linspace(0, 10, feature_dimension * 5)
    x2 = np.reshape(x2, [5, feature_dimension])
    x3 = np.linspace(0, 10, feature_dimension * 5) -5
    x3 = np.reshape(x3, [5, feature_dimension])
    x = np.vstack([x1, x2, x3])
    y = np.vstack([np.ones((5, 1))+np.ones((5,1)), np.ones((5, 1)), np.zeros((5, 1))])
    y = np.squeeze(y)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.one_hot(y, 3)
    return x, y


def construct_loss(pred, y, method='log_loss'):

    sample_num = obt_shape(y)[0]
    if method == 'log_loss':
        tmp = tf.multiply(y, tf.log(pred))
        # 拆分reduce_sum
        single_loss = -tf.reduce_sum(tmp, axis=1)
        loss = 1/sample_num*tf.reduce_sum(single_loss)
    else:
        raise Exception("不支持损失函数: {f} 计算".format(f=method))
    return loss


def softmax(x):

    exp_ele = tf.exp(-x)
    exp_sum = tf.reduce_sum(exp_ele, axis=1, keep_dims=True)
    exp_sum = tf.tile(exp_sum, [1, 3])
    softmax_result = tf.divide(exp_ele, exp_sum)
    return softmax_result


def softmax_regression():
    """
    与逻辑回归不同的是，softmax回归用于多分类
    1. 权重矩阵从[1, feature_dimension]将变成[target_num, feature_dimension]
    2. 偏置从单值变成[1, target_num]
    3. 损失函数 -> 交叉熵损失函数, 对于标签为0维度的损失将忽略。
    4. 标签将变为one hot形式
    :return:
    """
    target_num = 3
    feature_dimension = 2
    x, y = prepare_data(feature_dimension)
    w = tf.Variable(
        tf.truncated_normal(shape=[target_num, feature_dimension], mean=0.0, stddev=0.2, seed=2020, dtype=tf.float32))
    # 超参数
    learning_rate = 0.001
    loop = 1000
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 参数更新逻辑
    for l in range(loop):
        # 当前预测值
        pred = tf.matmul(x, tf.transpose(w))
        pred = softmax(pred)
        # 当前损失函数
        cur_loss = construct_loss(pred, y)
        # 参数更新计算方法
        # 1. 标签与预测值相减
        tmp = tf.subtract(y, tf.multiply(pred, y))
        # 2. 在target维度split，得到list
        each_class_target_loss = tf.split(tmp, target_num, 1)
        # 3. 声明参数更新矩阵
        update_w = tf.zeros([target_num, feature_dimension])
        # 4. 针对每个target，对w的每个target(也就是每行)进行更新
        for i, data in enumerate(each_class_target_loss):
            class_index = i
            data_expand = tf.tile(data, [1, feature_dimension])
            update_vector = tf.reduce_sum(tf.multiply(data_expand, x), axis=0)
            update_vector = tf.expand_dims(update_vector, 0)
            cur_update_w = tf.pad(update_vector, [[class_index, target_num - 1 - class_index], [0, 0]])
            update_w = tf.add(cur_update_w, update_w)
        w = w - tf.multiply(learning_rate, update_w)
    sess.close()

if __name__ == '__main__':
    softmax_regression()