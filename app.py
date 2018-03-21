#coding=utf-8

import tensorflow as tf

import mnist

import picTransform as pt

image= pt.tranform("one.png")
logits = mnist.inference(image, 128, 32)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "myMnistNet/save_net.ckpt")
result = sess.run(tf.nn.softmax(logits))
predictNum=result[0].tolist()
print(predictNum)
print(predictNum.index(max(predictNum)))





