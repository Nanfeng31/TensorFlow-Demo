# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:19:58 2020

@author: xxdn
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras

class Model:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape = (28, 28)),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dense(10, activation='softmax')
            ])
    """
    keras.layers.Dense(units, 
				  activation=None, 
				  use_bias=True, 
				  kernel_initializer='glorot_uniform', 
				  bias_initializer='zeros', 
				  kernel_regularizer=None, 
				  bias_regularizer=None, 
			      activity_regularizer=None, 
				  kernel_constraint=None, 
				  bias_constraint=None)
    
    """
    
    #self.keras.layers.