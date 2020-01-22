# coding: utf-8
# @Time    : 2020/1/20 11:42 上午
# @Author  : douwentao
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def apply_rnn(X, time_steps, state_size, batch_size, weight, bias):
    # 1. 在时间轴分割输入数据
    X_in = tf.split(X, axis=1, num_or_size_splits=time_steps)
    for i in range(len(X_in)):
        X_in[i] = tf.squeeze(X_in[i])
    # 2. 定义RNN细胞结构
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    # 3. 初始化状态
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # 4. 定义RNN链式结构, outputs为每个时刻的输出列表, final_states为最后时刻输出的状态
    outputs, final_states = tf.nn.static_rnn(cell, X_in, initial_state=init_state, dtype=tf.float32)
    # 5. 将最后时刻的状态转换成对应输出维度
    final_output = tf.add(tf.matmul(final_states, weight), bias)
    # 6. 利用softmax将输出转换为概率
    final_prob = tf.nn.softmax(final_output)
    return final_prob


def rnn_test():
    # 参数设置
    batch_size = 128
    in_num = 28
    time_steps = 28
    state_size = 128
    num_classes = 10
    learning_rate = 0.001
    # 输入数据占位符
    X = tf.placeholder(shape=[batch_size, time_steps, in_num], dtype=tf.float32)
    Y = tf.placeholder(shape=[batch_size, num_classes], dtype=tf.int32)
    # 定义rnn内部权重和偏置
    weight = tf.Variable(tf.truncated_normal(shape=[state_size, num_classes], mean=0.0, stddev=0.02, dtype=tf.float32))
    bias = tf.Variable(tf.truncated_normal(shape=[num_classes], mean=0.0, stddev=0.02, dtype=tf.float32))
    # 得到一个批次的样本的概率, [batch_size, num_classes]
    final_prob = apply_rnn(X, time_steps, state_size, batch_size, weight, bias)
    # 预测概率和样本标签均为one-hot形式，计算识别率
    correct_pred = tf.equal(tf.argmax(final_prob, axis=1), tf.argmax(Y, axis=1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # 计算一个批次中每个样本的损失
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=final_prob, labels=Y)
    # 对一个批次中所有样本的损失进行求和
    total_loss = tf.reduce_mean(loss)
    # 定义优化方法，并反向传播最小化损失
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            if i != 0 and i % 100 == 0:
                print("当前batch识别率: ", sess.run(acc, feed_dict={X: batch_xs, Y: batch_ys}))
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, time_steps, in_num])
            _ = sess.run([train_op], feed_dict={X: batch_xs, Y: batch_ys})

if __name__ == '__main__':
    rnn_test()











