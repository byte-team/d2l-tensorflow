# coding: utf-8
# @Time    : 2020/1/17 12:25 下午
# @Author  : douwentao
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