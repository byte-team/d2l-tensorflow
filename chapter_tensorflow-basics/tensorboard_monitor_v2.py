# coding: utf-8
# @Time    : 2020/2/1 1:46 下午
# @Author  : douwentao
# Tensorboard还支持SCALARS, IMAGES, AUDIO, DISTRIBUTIONS, HISTOGRAMS, TEXT等6种界面
# 其中DISTRIBUTIONS和HISTOGRAMS两栏的数据源是相同的(tf.summary.histogram)，只是表达形式不同
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_inference import inference_with_tensorboard, inference

SUMMARY_DIR = "./tensorboard/monitor_v2/"
BATCH_SIZE = 100
TRAIN_STEPS = 30000


class MnistModel(object):
    def __init__(self, mnist):
        self.mnist = mnist
        self.regularization_rate = 0.0001
        self.mv_avg_decay = 0.99
        self.learning_rate_base = 0.8
        self.learning_rate_decay = 0.99
        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)
        # 定义输入
        with tf.name_scope('input'):
            self.x = tf.placeholder(shape=[None, 784], name='x-input', dtype=tf.float32)
            self.y_ = tf.placeholder(shape=[None, 10], name='y-input', dtype=tf.float32)
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)
        self.y = inference_with_tensorboard(self.x, regularizer)
        # self.y = inference(self.x, regularizer)
        self.global_step = tf.Variable(0, trainable=False)
        # 定义滑动平局模型
        variable_averages = tf.train.ExponentialMovingAverage(
            self.mv_avg_decay,
            self.global_step
        )
        # 针对所有可训练变量进行滑动平均
        variables_average_op = variable_averages.apply(tf.trainable_variables())
        # 损失计算
        with tf.name_scope('cross_entropy_loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.y,
                    labels=tf.argmax(self.y_, 1))
            )
            self.loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('total_loss', self.loss)

        with tf.name_scope('train_step'):
            # staircase参数
            learning_rate = tf.train.exponential_decay(
                self.learning_rate_base,
                self.global_step,
                self.mnist.train.num_examples / BATCH_SIZE,
                self.learning_rate_decay
            )
            self.train_steps = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(self.loss,                                                                global_step=self.global_step)
            with tf.control_dependencies([self.train_steps, variables_average_op]):
                self.train_op = tf.no_op(name='train')
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_predictons'):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, dtype=tf.float32)
                    )
                    tf.summary.scalar('accuracy', self.accuracy)
        # 整合所有summary
        self.merged = tf.summary.merge_all()
        pass


def train_mnist_with_tensorboard():
    mnist = input_data.read_data_sets('../chapter_recurrent-neural-networks/MNIST_data', one_hot=True)
    model = MnistModel(mnist)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            SUMMARY_DIR, sess.graph
        )
        sess.run(tf.global_variables_initializer())
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _, total_loss, acc = sess.run(
                [model.merged, model.train_op, model.loss, model.accuracy],
                feed_dict={
                    model.x: xs,
                    model.y_: ys
                })
            summary_writer.add_summary(summary, i)
            if i % 1000 == 0:
                print("step: {step}, loss:{loss}, acc: {acc}".format(
                    step=i,
                    loss=total_loss,
                    acc=acc
                ))
    summary_writer.close()


def main():
    train_mnist_with_tensorboard()


if __name__ == '__main__':0
TRAIN_STEPS = 30000


class MnistModel(object):
    def __init__(self, mnist):
        self.mnist = mnist
        self.regularization_rate = 0.0001
        self.mv_avg_decay = 0.99
        self.learning_rate_base = 0.8
        self.learning_rate_decay = 0.99
        regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)
        # 定义输入
        with tf.name_scope('input'):
            self.x = tf.placeholder(shape=[None, 784], name='x-input', dtype=tf.float32)
            self.y_ = tf.placeholder(shape=[None, 10], name='y-input', dtype=tf.float32)
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)
        self.y = inference_with_tensorboard(self.x, regularizer)
        # self.y = inference(self.x, regularizer)
        self.global_step = tf.Variable(0, trainable=False)
        # 定义滑动平局模型
        variable_averages = tf.train.ExponentialMovingAverage(
            self.mv_avg_decay,
            self.global_step
        )
        # 针对所有可训练变量进行滑动平均
        variables_average_op = variable_averages.apply(tf.trainable_variables())
        # 损失计算
        with tf.name_scope('cross_entropy_loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.y,
                    labels=tf.argmax(self.y_, 1))
            )
            self.loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('total_loss', self.loss)

        with tf.name_scope('train_step'):
            # staircase参数参考optimize_methods.ipynb
            learning_rate = tf.train.exponential_decay(
                self.learning_rate_base,
                self.global_step,
                self.mnist.train.num_examples / BATCH_SIZE,
                self.learning_rate_decay,
                staircase=True
            )
            self.train_steps = tf.train.GradientDescentOptimizer(
                learning_rate
            ).minimize(self.loss,                                                                global_step=self.global_step)
            with tf.control_dependencies([self.train_steps, variables_average_op]):
                self.train_op = tf.no_op(name='train')
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_predictons'):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(
                        tf.cast(correct_prediction, dtype=tf.float32)
                    )
                    tf.summary.scalar('accuracy', self.accuracy)
        # 整合所有summary
        self.merged = tf.summary.merge_all()
        pass


def train_mnist_with_tensorboard():
    mnist = input_data.read_data_sets('../chapter_recurrent-neural-networks/MNIST_data', one_hot=True)
    model = MnistModel(mnist)
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            SUMMARY_DIR, sess.graph
        )
        sess.run(tf.global_variables_initializer())
        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _, total_loss, acc = sess.run(
                [model.merged, model.train_op, model.loss, model.accuracy],
                feed_dict={
                    model.x: xs,
                    model.y_: ys
                })
            summary_writer.add_summary(summary, i)
            if i % 1000 == 0:
                print("step: {step}, loss:{loss}, acc: {acc}".format(
                    step=i,
                    loss=total_loss,
                    acc=acc
                ))
    summary_writer.close()


def main():
    train_mnist_with_tensorboard()


if __name__ == '__main__':
    main()
