#coding=utf-8

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
import mnist
import picTransform as pt

imageForTest= pt.tranform("one.png")

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder



def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(100, False)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def run_training():
    data_sets = read_data_sets('/tmp/tensorflow/mnist/input_data', False)
    with tf.Graph().as_default():

        images_placeholder, labels_placeholder = placeholder_inputs(100)  # 每100张送进去计算一次所以这里的向量维度100

        logits = mnist.inference(images_placeholder, 128, 32)

        loss = mnist.loss(logits, labels_placeholder)

        train_op = mnist.training(loss, 0.01)

        summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()  #前面都是定义变量，这里对该图中的变量进行初始化

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter("logs/", sess.graph)

        sess.run(init)


        for step in xrange(2000): # 每一步取100张，一共2000部，取20万张

            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            #sess.run（）函数对loss进行计算，得到lossValue用于查看拟合度是否上身，主要的计算在于执行train_op，返回值_表示匿名，既不关心（应为train_op只是一个计算过程，所以没有返回实体，所以打印并没有区别）
            if step % 100 == 0:
                print(loss_value)
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()


            #    print(train_op)
        save_path=saver.save(sess,"myMnistNet/save_net.ckpt")
        print(save_path)


def main(_):

    run_training()


if __name__ == '__main__':

    tf.app.run(main=main)