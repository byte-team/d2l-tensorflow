# coding: utf-8
# @Time    : 2020/1/26 9:53 下午
# @Author  : douwentao
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    x = tf.placeholder(shape=[None, mnist_inference.INPUT_NODE], name="x_input", dtype=tf.float32)
    y_ = tf.placeholder(shape=[None, mnist_inference.OUTPUT_NODE], name="y_input", dtype=tf.float32)

    validate_feed = {x: mnist.validation.images,
                     y_: mnist.validation.labels}

    y = mnist_inference.inference(x, None)

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(
        mnist_train.MOVING_AVERAGE_DECAY
    )
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy = sess.run(acc, feed_dict=validate_feed)
                print("{global_step} step accuracy {acc}".format(
                    global_step=global_step,
                    acc=accuracy
                ))
            else:
                print("no checkpoint")
            time.sleep(EVAL_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets('../chapter_recurrent-neural-networks/MNIST_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
