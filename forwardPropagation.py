# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:44:49 2020

@author: Nanfeng
"""
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

weight_1 = tf.Variable(tf.random_normal((2, 3), stddev=1 , seed=1))
weight_2 = tf.Variable(tf.random_normal((3, 1), stddev=1 , seed=1))

#this is how to define a 1x2 matrix
# x = tf.constant([[0.7, 0.9]])
# and this is how to batch a collection
x = tf.placeholder(tf.float32, shape=(3,2), name="input")
# y_ = tf.placeholder(tf.float32, shape=(3,2), name="input")
a = tf.matmul(x, weight_1)
y = tf.matmul(a, weight_2)
session = tf.Session()
# It's important to inite all variables
init_op = tf.global_variables_initializer()
session.run(init_op)
print(session.run(y, feed_dict={x:[[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

# y = tf.sigmoid(y)
# cross_entropy = -tf.reduce_mean(
#     y_ * tf.log(tf.clip_by_value(y, 1e-10))
#     + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))) 
# learning_rate = 0.001
# tran_step=\
#     tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
