# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:38:01 2020

@author: Nanfeng
"""
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
weight_1 = tf.Variable(tf.random_normal((2, 3), stddev=1 , seed=1))
weight_2 = tf.Variable(tf.random_normal((3, 1), stddev=1 , seed=1))

#this is how to define a 1x2 matrix
x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, weight_1)
y = tf.matmul(a, weight_2)

seeion = tf.Session()
seeion.run(weight_1.initializer)
seeion.run(weight_2.initializer)
#print(weight_1 , "and" , weight_2)
print("weight_1:", seeion.run(weight_1))
print("weight_2:", seeion.run(weight_2))
print("final", seeion.run(y))
# tf.assign(weight_1, weight_2, validate_shape=False)
# print("weight_1:", seeion.run(weight_1))
# print("weight_2:", seeion.run(weight_2))
seeion.close

x1 = tf.placeholder(tf.float32, shape=(1,2), name="input")

seeion1 = tf.Session()
init_op = tf.global_variables_initializer()
seeion1.run(init_op)
print("final", seeion.run(y, feed_dict={x1:[[0.7, 0.9]]}))
seeion1.close