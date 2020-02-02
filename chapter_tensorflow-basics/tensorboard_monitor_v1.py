# coding: utf-8
# @Time    : 2020/2/1 1:02 下午
# @Author  : douwentao
# 1. 在Tensorboard中, 同一个命名空间下的所有节点会被缩略成一个节点。
# 2. 计算图可视化:
#     writer = tf.summary.FileWriter("path", tf.get_default_graph())
#     writer.close()
# 3. 除了计算图的结构，Tensorboard还可以展示节点的基本信息和运行时消耗的时间和空间。
#    同时包含device选项，可以用来看那个模型或者节点使用了某个GPU还是使用了某个CPU
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_train


def train_mnist_with_tensorboard():
    # 训练后,在当前目录下,输入 tensorboard --logdir=./ 即可进入tensorboard
    SUMMARY_DIR = "./tensorboard/monitor_v1/"

    mnist = input_data.read_data_sets('../chapter_recurrent-neural-networks/MNIST_data', one_hot=True)
    model = mnist_train.MnistModel(mnist)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 一般会在初始化后创建writer
        train_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        for i in range(mnist_train.TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(mnist_train.BATCH_SIZE)
            # 1000轮记录一次运行状态
            if i % 1000 == 0:
                # 配置运行时需要记录的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto
                run_metadata = tf.RunMetadata()
                # 配置信息和记录运行信息的proto作为参数传入run函数中，
                # run_metadata记录下当前会话运行时的信息
                _, loss_value, step = sess.run([model.train_op,
                                                model.loss,
                                                model.global_step],
                                               feed_dict={model.x: xs,
                                                          model.y_: ys},
                                               options=run_options,
                                               run_metadata=run_metadata)
                # run_metadata记录下信息后，添加到train_writer中
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                print("step: {step}, loss: {loss}".format(
                    step=step,
                    loss=loss_value
                ))
            else:
                _, loss_value, step = sess.run([model.train_op,
                                                model.loss,
                                                model.global_step],
                                               feed_dict={model.x: xs,
                                                          model.y_: ys})


def main():
    train_mnist_with_tensorboard()


if __name__ == '__main__':
    main()
