#coding=utf-8
import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images,hidden1_units,hidden2_units):
    print('Training Data Eval:')
    with tf.name_scope("hidden1"):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name = 'weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),name = 'biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope("hidden2"):
        weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],stddev=1.0 / math.sqrt(float(hidden1_units))),
        name = 'weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
        name = 'biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):#最后一层输出层使用softmax来归一化和概率分布
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases #将最后一个隐藏层结合输出，用于等下计算loss，以及进入softmax进行计算
    return logits


def loss(logits, labels):

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
