# coding: utf-8
# @Time    : 2020/1/22 4:16 下午
# @Author  : douwentao
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def apply_static_lstm(X, in_num, weight_lstm, bias_lstm, time_steps, state_size, batch_size):
    # [batch_size, time_steps, in_num] -> [batch_size*time_steps, in_num]
    X_in = tf.reshape(X, [-1, in_num])
    # [batch_size*time_steps, in_num] -> [batch_size*time_steps, state_size]
    X_in = tf.matmul(X_in, weight_lstm['in']) + bias_lstm['in']
    # [batch_size*time_steps, state_size] -> [batch_size, time_steps, state_size]
    X_in = tf.reshape(X_in, [-1, time_steps, state_size])
    # 按照第一维分割为列表
    X_in = tf.split(X_in, axis=1, num_or_size_splits=time_steps)
    for i in range(len(X_in)):
        X_in[i] = tf.squeeze(X_in[i])
    # 定义lstm细胞
    cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, forget_bias=1.0)
    # 定义batch_size个初始状态
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # 利用输入列表循环更新细胞参数
    outputs, final_states = tf.nn.static_rnn(cell, X_in, initial_state=init_state, dtype=tf.float32)
    final_output = tf.add(tf.matmul(final_states[1], weight_lstm['out']), bias_lstm['out'])
    final_prob = tf.nn.softmax(final_output)
    return final_prob


def apply_dynamic_lstm(X, in_num, weight_lstm, bias_lstm, time_steps, state_size, batch_size):
    #   dynamic_rnn
    #   1. 自动跳动batch内部padding位置的运算;
    #   2. batch之间的序列长度可不相同;
    #   3. 如果注意在计算损失的时候，需要将padding位置的权重设置为0，以避免无效位置的预测干扰模型的训练

    # [batch_size, time_steps, in_num] -> [batch_size*time_steps, in_num]
    X_in = tf.reshape(X, [-1, in_num])
    # [batch_size*time_steps, in_num] -> [batch_size*time_steps, state_size]
    X_in = tf.matmul(X_in, weight_lstm['in']) + bias_lstm['in']
    # [batch_size*time_steps, state_size] -> [batch_size, time_steps, state_size]
    X_in = tf.reshape(X_in, [-1, time_steps, state_size])
    # 定义lstm细胞
    cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, forget_bias=1.0)
    # 定义batch_size个初始状态
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    # 利用输入列表循环更新细胞参数
    outputs, final_states = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, dtype=tf.float32, time_major=False)
    # 1. 使用单层lstm最后一个输出
    final_output = tf.add(tf.matmul(final_states[1], weight_lstm['out']), bias_lstm['out'])
    final_prob = tf.nn.softmax(final_output)
    return final_prob


def apply_bid_dynamic_lstm(X, in_num, weight_bid_lstm, bias_bid_lstm, time_steps, state_size, batch_size):
    # [batch_size, time_steps, in_num] -> [batch_size*time_steps, in_num]
    X_in = tf.reshape(X, [-1, in_num])
    # [batch_size*time_steps, in_num] -> [batch_size*time_steps, state_size]
    X_in = tf.matmul(X_in, weight_bid_lstm['in']) + bias_bid_lstm['in']
    # [batch_size*time_steps, state_size] -> [batch_size, time_steps, state_size]
    X_in = tf.reshape(X_in, [-1, time_steps, state_size])
    forward_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
    backward_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
    forward_init_state = forward_cell.zero_state(batch_size, dtype=tf.float32)
    backward_init_state = backward_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                      backward_cell,
                                                      X_in,
                                                      initial_state_fw=forward_init_state,
                                                      initial_state_bw=backward_init_state)

    outputs = tf.concat(outputs, axis=2)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]

    final_output = tf.add(tf.matmul(outputs, weight_bid_lstm['out']), bias_bid_lstm['out'])
    final_prob = tf.nn.softmax(final_output)

    return final_prob


def apply_multi_dynamic_lstm(X, in_num, weight_lstm, bias_lstm, time_steps, state_size, batch_size, num_layers):
    # lstm的dropout方法一般只在于不同层循环体结构之间使用，而不在同一层的循环体结构之间。
    # [batch_size, time_steps, in_num] -> [batch_size*time_steps, in_num]
    X_in = tf.reshape(X, [-1, in_num])
    # [batch_size*time_steps, in_num] -> [batch_size*time_steps, state_size]
    X_in = tf.matmul(X_in, weight_lstm['in']) + bias_lstm['in']
    # [batch_size*time_steps, state_size] -> [batch_size, time_steps, state_size]
    X_in = tf.reshape(X_in, [-1, time_steps, state_size])

    single_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell
    # 堆叠多层lstm细胞, 并在每个细胞之间增加dropout方法
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
        [
            tf.nn.rnn_cell.DropoutWrapper(
                single_lstm_cell(state_size),
                input_keep_prob=1.0,
                output_keep_prob=1.0,
                state_keep_prob=1.0
            ) for _ in range(num_layers)
        ]
    )
    # 声明初始状态
    stacked_lstm_init_state = stacked_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn处理方式相当于以下代码
    # for i in range(time_steps):
    #     if i > 0: tf.get_variable_scope().reuse_variables()
    #     stacked_lstm_output, state = stacked_lstm_cell(cur_input, state)
    #     final_output = fully_connected(stacked_lstm_output)
    #     loss += calc_loss(final_output, expect_output)
    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm_cell,
                                   X,
                                   initial_state=stacked_lstm_init_state,
                                   dtype=tf.float32)
    outputs = outputs[:, -1, :]
    final_output = tf.add(tf.matmul(outputs, weight_lstm['out']), bias_lstm['out'])
    final_prob = tf.nn.softmax(final_output)
    return final_prob

def lstm_test():
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
    # 定义lstm内部权重和偏置
    weight_lstm = {
        'in': tf.Variable(tf.random_normal(shape=[in_num, state_size])),
        'out': tf.Variable(tf.random_normal(shape=[state_size, num_classes]))
    }
    bias_lstm = {
        'in': tf.Variable(tf.random_normal(shape=[state_size, ])),
        'out': tf.Variable(tf.random_normal(shape=[num_classes, ]))
    }
    weight_bid_lstm = {
        'in': tf.Variable(tf.random_normal(shape=[in_num, state_size])),
        'out': tf.Variable(tf.random_normal(shape=[2*state_size, num_classes]))
    }
    bias_bid_lstm = {
        'in': tf.Variable(tf.random_normal(shape=[state_size, ])),
        'out': tf.Variable(tf.random_normal(shape=[num_classes, ]))
    }
    # -------------------------------------------------------------
    # 得到一个批次的样本的概率
    # -------------------------------------------------------------
    # 1. 使用静态lstm, 即时序长度固定
    # final_prob = apply_static_lstm(X, in_num, weight_lstm, bias_lstm, time_steps, state_size, batch_size)
    # -------------------------------------------------------------
    # 2. 使用动态lstm, 即时序长度可以变化
    # final_prob = apply_dynamic_lstm(X,in_num, weight_lstm, bias_lstm, time_steps, state_size, batch_size)
    # -------------------------------------------------------------
    # 3. 使用双向lstm
    # final_prob = apply_bid_dynamic_lstm(X,
    #                                     in_num,
    #                                     weight_bid_lstm,
    #                                     bias_bid_lstm,
    #                                     time_steps,
    #                                     state_size,
    #                                     batch_size)
    # -------------------------------------------------------------
    # 4. 使用多层堆叠的lstm
    num_layers = 2
    final_prob = apply_multi_dynamic_lstm(X,
                                          in_num,
                                          weight_lstm,
                                          bias_lstm,
                                          time_steps,
                                          state_size,
                                          batch_size,
                                          num_layers)
    # ----------------------------------------------------------

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
    lstm_test()
