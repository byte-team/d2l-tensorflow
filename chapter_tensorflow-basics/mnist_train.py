# coding: utf-8
# @Time    : 2020/1/26 7:36 下午
# @Author  : douwentao
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型存储路径
MODEL_SAVE_PATH = "./checkpoints"
MODEL_NAME = "model.ckpt"


def train(mnist):

    x = tf.placeholder(
        shape=[None, mnist_inference.INPUT_NODE],
        name="x_input",
        dtype=tf.float32
    )
    y_ = tf.placeholder(
        shape=[None, mnist_inference.OUTPUT_NODE],
        name="y_input",
        dtype=tf.float32
    )
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    # ---------------------滑动平均---------------------------
    # 定义滑动平均模型
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY,
        global_step
    )
    # 针对所有可训练变量进行滑动平均
    variables_average_op = variable_averages.apply(tf.trainable_variables())
    # --------------------------------------------------------
    # logits为one-hot形式, y为连续值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,
        labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # ---------------------学习率指数级下降---------------------------
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    # --------------------------------------------------------
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step
    )
    # 只执行train_step和variables_average_op
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name='train')

    # max_to_keep默认参数为5
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op,
                                            loss,
                                            global_step],
                                           feed_dict={
                                               x: xs,
                                               y_: ys
                                           })
            if i % 1000 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)
                print("after {step} training steps, model has been saved".format(
                    step=step
                ))
                print("loss: ", loss_value)


def main():
    mnist = input_data.read_data_sets('../chapter_recurrent-neural-networks/MNIST_data', one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()