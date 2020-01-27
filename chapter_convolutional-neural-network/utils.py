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
        weights = tf.get_variable("weights", shape=[in_num, out_num], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[out_num], dtype=tf.float32)
        act = tf.nn.xw_plus_b(x, weights, bias, name=scope.name)
        # 返回是否使用激活函数的结果
        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


# BN 操作  -> y = scale * (x - mean) / var + shift, 均值与方差为
# 当前batch的每个channel的均值与方差。Tensorflow中的均值与方差为滑动平均后的值。
# 训练完成后，进行推理时，使用训练后的滑动平均后的均值与方差对feature_map进行标准化操作

# BN 通过标准化，将非线性激活函数的输入值规范为均值为0，方差为1的高斯分布
# 但是由于大多输入值在0附近，导致非线性表达能力差(sigmoid函数在0值附近接近线性)
# 利用可训练的scale和shift参数，将非线性激活函数的输入值进行缩放或平移
# 能够使得既保证了梯度能足够大（防止梯度消失）,又能够保留非线性表达能力。

# BN优点
# 1. 可以使用较高的学习率，提高模型的训练速度，并且可以有效的避免梯度消失和梯度爆炸
# （在不使用batch normalization时，因为当学习率较大，模型参数变化快，每个神经元激活函数的输入值变化很快，导致同一层的神经元在不同批次下的
# 输入值的分布不相同，导致训练很慢。加入batch normalization后, 因为每个神经元激活函数前会对数据进行归一化操作，将数据的分布拉回
#  高斯分布，使得每个批次的数据分布大致相同，使得训练加快。）
# 2. 对模型参数的初始化要求不高
# (与1中的解释类似，由于每层进行了batch normalization，不管模型参数如何剧烈变化，神经元的输入值分布都较为稳定。)
# 3. 抑制过拟合，降低dropout使用，提高泛化能力
# (batch normalization是一种数据的归一化方法，归一化的作用就是消除奇异值对模型的影响，防止过拟合)

# BN缺点
# 1. BN 依赖batch_size, 当batch_size小的时候, BN效果不佳；当batch_size大的时候，显存占用多;
# 2. BN 不适用于序列化数据的网络

# 全连接层增加batch_normalization
def batch_norm_fully_connected(x, in_num, out_num, name, relu=True, is_training=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", shape=[in_num, out_num], dtype=tf.float32)
        bias = tf.get_variable("bias", shape=[out_num], dtype=tf.float32)
        act = tf.nn.xw_plus_b(x, weights, bias, name=scope.name)

        # 定义batch normalization需要学习的参数
        # gamma(缩放) 和 beta(平移) 都是可学习的变量
        gamma = tf.get_variable("gamma", shape=[out_num], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", shape=[out_num], initializer=tf.zeros_initializer())
        # shadow_mean 和 shadow_var为不可学习变量
        shadow_mean = tf.get_variable("shadow_mean",
                                      shape=[out_num],
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        shadow_var = tf.get_variable("shadow_var",
                                     shape=[out_num],
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
        # 进行归一化时，防止分母中的标准差接近0而得到nan值
        EPSILON = 1e-3
        MOVING_AVG_DECAY = 0.99

        def batch_norm_training():
            batch_mean, batch_var = tf.nn.moments(x, [0])
            # 计算mean和var的滑动平均值
            train_mean = tf.assign(shadow_mean, MOVING_AVG_DECAY * shadow_mean + (1 - MOVING_AVG_DECAY) * batch_mean)
            train_var = tf.assgin(shadow_var, MOVING_AVG_DECAY * shadow_var + (1 - MOVING_AVG_DECAY) * batch_var)
            # 控制mean和var的滑动平均值计算完成后才能计算batch normalization
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, EPSILON)

        def batch_norm_inference():
            return tf.nn.batch_normalization(x, shadow_mean, shadow_var, beta, gamma, EPSILON)

        batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
        if relu:
            return tf.nn.relu(batch_normalized_output)
        else:
            return batch_normalized_output


# 卷积层增加batch normalization
def batch_norm_conv(x, filter_height, filter_width, num_filters,
                    stride_y, stride_x, name, padding='SAME', is_training=True):
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

        gamma = tf.get_variable(name="gamma", shape=[num_filters], initializer=tf.ones_initializer())
        beta = tf.get_variable(name="beta", shape=[num_filters], initializer=tf.zeros_initializer())

        shadow_mean = tf.get_variable(name="shadow_mean",
                                      shape=[num_filters],
                                      trainable=False)
        shadow_var = tf.get_variable(name="shadow_var",
                                     shape=[num_filters],
                                     trainable=False)
        EPSILON = 1e-3
        MOVING_AVG_DECAY = 0.99

        def batch_norm_training():
            # 此时设定通道维度在第4维度, 得到的batch_mean与batch_var为一维向量，其长度与通道数量相同
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
            train_mean = tf.assign(shadow_mean, MOVING_AVG_DECAY * shadow_mean + (1 - MOVING_AVG_DECAY) * batch_mean)
            train_var = tf.assign(shadow_var, MOVING_AVG_DECAY * shadow_var + (1 - MOVING_AVG_DECAY) * batch_var)

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, EPSILON)

        def batch_norm_inference():
            return tf.nn.batch_normalization(x, shadow_mean, shadow_var, beta, gamma, EPSILON)

        batch_normalized_result = tf.cond(is_training, batch_norm_training, batch_norm_inference)
        return tf.nn.relu(batch_normalized_result)


# 池化层一般包括平均池化和最大池化，常采用的方法为最大池化
# 最大池化操作可以理解为降采样的过程，即在一个特定区域内提取其中的最大值作为该区域的特征
# 总结下优点:
# 1. 在计算机视觉领域，最大池化相当于特征提取的过程，其提取的特征具有: 平移，旋转，缩放不变形（脑补这三种操作，不管怎么搞，最大池化提取的特征都是相同的）
# 2. 在特征提取的过程中，同样降低了特征的维度，即剔除了杂质特征，选取了有用的显著的特征，有效的防止了过拟合，提升了模型的泛化能力，同时降低了计算复杂度
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


# 减少模型参数，降低过拟合，提升模型训练速度。
def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
