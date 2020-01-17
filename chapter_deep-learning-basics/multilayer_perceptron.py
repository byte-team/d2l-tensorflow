# coding: utf-8
# @Time    : 2020/1/17 2:37 下午
# @Author  : douwentao
import tensorflow as tf
from utils import obt_shape


class PerceptronLayer(object):

    """
    使用方法: 每层声明一个对象, 顺序前向传播，计算最后一层的误差矩阵, 倒序反向更新所有对象的参数
    """


    def __init__(self, input_feature_dimension, output_neuron_num, activator, learning_rate):
        """
        初始化
        :param input_feature_dimension: 输入特征维度
        :param output_neuron_num: 输出特征维度
        :param activator: 激活函数
        :param learning_rate: 学习率
        """
        self.input_feature_dimension = input_feature_dimension
        self.output_neuron_num = output_neuron_num
        # 初始化权重系数
        self.weight = tf.Variable(tf.truncated_normal(shape=[output_neuron_num, input_feature_dimension],
                                                      mean=0.0,
                                                      stddev=0.02,
                                                      seed=2020,
                                                      dtype=tf.float32))
        # 初始化偏置
        self.bias = tf.Variable(tf.truncated_normal(shape=[1, output_neuron_num],
                                                    mean=0.0,
                                                    stddev=0.02,
                                                    seed=2020,
                                                    dtype=tf.float32))
        # 定义当前层激活函数
        self.activator = activator
        self.learning_rate = learning_rate

    def forward_propagation(self, input):
        """
        卷积操作调用函数
        注意这里是inplace操作
        :param input: 输入
        :param kernel: 过滤器
        :param output: 输出（inplace）
        :param stride: 步长
        :param bias:
        :return:
        """
        self.input = input
        cur_pred = tf.matmul(input, tf.transpose(self.weight))
        self.output_before_activator = tf.add(cur_pred, self.bias)
        self.output_after_activator = self.activator.forward(self.output_before_activator)
        return self.output_after_activator

    def bp_sensitive_map(self, sensitive_map, activator):
        """
        返回上一层的sensitive_map, 形状为[batch, input_feature_dimension]
        :param sensitive_map:
        :param activator:
        :return:
        """
        sensitive_map_list = tf.split(sensitive_map, obt_shape(sensitive_map)[0], axis=0)
        for i, sm in enumerate(sensitive_map_list):
            cur_map = tf.matmul(tf.transpose(self.weight), tf.transpose(sm))
            sensitive_map_list[i] = cur_map
        input_list = tf.split(self.input, obt_shape(self.input)[0], axis=0)
        grad_list = []
        for i, single_input in enumerate(input_list):
            cur_map = tf.multiply(tf.transpose(single_input), self.activator.backward(sensitive_map_list[i]))
            cur_map = tf.squeeze(cur_map, axis=1)
            grad_list.append(cur_map)
        return tf.stack(grad_list, axis=0)


    def update_gradients(self, sensitive_map):
        # sensitive_map 维度 [batch, output_neuron_num]
        # 输入的维度 [batch, input_feature_dimension], batch也可能为1
        # 每个样本更新一次 [N, output_neuron_num, input_feature_dimension]
        # 1. 首先对输入进行split, 这里的输入是由forward_propagation函数传进来的
        input_list = tf.split(self.input, obt_shape(self.input)[0], 0)
        sensitive_map_list = tf.split(sensitive_map, obt_shape(self.input)[0], 0)
        for i, input in enumerate(input_list):
            cur_update_weight = tf.matmul(tf.transpose(sensitive_map_list[i]), input)
            input_list[i] = cur_update_weight
        # 平均操作
        cur_batch_update_weight = tf.stack(input_list, axis=0)
        self.update_weight = tf.reduce_mean(cur_batch_update_weight, axis=0)
        self.update_bias = tf.reduce_mean(sensitive_map, axis=0)
        self.weight = self.weight - tf.multiply(self.learning_rate, self.update_weight)
        self.bias = self.bias - tf.multiply(self.learning_rate, self.update_bias)
        return

    def backward_propagation(self, sensitive_map, activator):
        """
        返回上一层的误差矩阵供上一层更新参数使用
        :param sensitive_map: 当前层的误差矩阵
        :param activator: 使用的激活函数
        :return:
        """
        # 前向传播一次
        self.forward_propagation(input)
        # 计算前一层误差项
        last_sensitive_map = self.bp_sensitive_map(sensitive_map, activator)
        # 更新当前卷积核权重及偏置的导数
        self.update_gradients(sensitive_map)

        return last_sensitive_map
