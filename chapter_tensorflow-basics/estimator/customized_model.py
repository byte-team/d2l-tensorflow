# coding: utf-8
# @Time    : 2020/2/5 7:05 下午
# @Author  : douwentao

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

def lenet(x, is_training):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(net, 64, 3, activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 1024)
    net = tf.layers.dropout(net, rate=0.4, training=is_training)
    return tf.layers.dense(net, 10)

# 定义的函数有4个输入:
# 1. features给出了在输入函数中会提供的输入层张量
# (这是一个字典里的内容是通过tf.estimator.inputs.numpy_input_fn中的x参指定的）
# 2. labels是标签, 这个字段内容是通过numpy_input_fn中y参数给出的。
# 3. mode的取值有3种可能，分别对应Estimator类的train, evaluate和predict这3个函数
# （通过这个参数可以判断当前是否为训练过程）
# 4. 最后一个参数params是一个字典，模型超参数(例如学习率)
def model_fn(features, labels, mode, params):
    predict = lenet(features["image"], mode==tf.estimator.ModeKeys.TRAIN)
    if mode == tf.estimator.ModeKeys.PREDICT:
        # 使用EstimatorSpec传递返回值, 并通过predictions参数指定返回的结果
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"result": tf.argmax(predict, 1)}
        )
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predict,
            labels=labels
        )
    )
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"]
    )
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )
    eval_metrics_ops = {
        "acc": tf.metrics.accuracy(
            tf.argmax(predict, 1),
            labels
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics_ops
    )

mnist = input_data.read_data_sets('../../chapter_recurrent-neural-networks/MNIST_data', one_hot=False)
model_params = {
    "learning_rate": 0.01
}
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    params=model_params
)

# 输入
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"image": mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True
)
# 训练
estimator.train(input_fn=train_input_fn)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"images": mnist.test.images},
    y=mnist.test.labels.astype(np.int32),
    num_epochs=1,
    batch_size=128,
    shuffle=False
)
test_results = estimator.evaluate(
    input_fn=test_input_fn
)
acc = test_results["acc"]
print("acc: ", acc)