# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:21:22 2020

@author: Nanfeng
"""
import tensorflow.compat.v1 as tf
from numpy.random import RandomState

tf.disable_eager_execution()
# define size of train set
batch_size = 8;
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# This floor has 3 node, placeSettting
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# Bipartisan model only one output true/false
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# forward propagation
a = tf.matmul(x, w1)
y = tf.matmul(a ,w2)

# defin lossfunction and 
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))) 
learning_rate = 0.001
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# create a random trainset
rng = RandomState(1)
dataset_size = 128
X = rng.rand(dataset_size, 2)

# define: x1 + x2 <1 positive
Y = [[int(x1 + x2) < 1] for (x1, x2) in X]
# Y = [[int(x1 + x2)] for (x1, x2) in X]
# create a session for run
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    '''
    how many times you need to train
        each times get one batch from traindata_set
    '''
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # to make sure not to overflow
        # send symbols to x by different times
        sess.run(train_step,
                 feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x:X, y_:Y})
            print("After %d training step(s), cross entropy on all data is %g"%
                  (i ,total_cross_entropy))
            pass
        pass
    print(sess.run(w1))
    print(sess.run(w2))