# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:34:25 2020

@author: Nanfeng
"""

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a_vector = tf.constant([1.0, 2.0], name = "a")
b_vector = tf.constant([2.0, 3.0], name = "b")
result = a_vector + b_vector
print(a_vector.graph is tf.compat.v1.get_default_graph())

graph1 = tf.Graph();
with graph1.as_default():
    v = tf.compat.v1.get_variable(
        "v", initializer = tf.zeros([1]))
    pass

graph2 = tf.Graph();
with graph2.as_default():
    v = tf.compat.v1.get_variable(
        "v", initializer = tf.ones([1]))
    pass

with tf.compat.v1.Session(graph = graph1) as sess:
    tf.compat.v1.global_variables_initializer().run()
    with tf.compat.v1.variable_scope("", reuse = True):
        print(sess.run(tf.compat.v1.get_variable("v")))
        pass
    pass

with tf.compat.v1.Session(graph = graph2) as sess:
    tf.compat.v1.global_variables_initializer().run()
    with tf.compat.v1.variable_scope("", reuse = True):
        print(sess.run(tf.compat.v1.get_variable("v")))
        pass
    pass

