# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:50:12 2020

@author: Nanfeng
"""
import tensorflow.compat.v1 as tf

a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")
result = tf.add(a ,b, name = "add")
print("directly :result" , result)
print("session.run", tf.Session().run(result))
print("eval(session = Session)" , result.eval(session = tf.Session()))
tf.Session().close

with tf.Session() as session:
    session.run(result)
    print("+++" , result.eval())
    pass
with tf.Session().as_default():
    print("para_eval" , result.eval)
    print("para_eval()" , result.eval())
    pass
sess = tf.InteractiveSession()
print("tf.interactiveSession()" , sess.run(result))
sess.close

config = tf.ConfigProto(allow_soft_placement = True,
                        log_device_placement = True)
sess1 = tf.InteractiveSession(config = config)
sess2 = tf.Session(config = config)
