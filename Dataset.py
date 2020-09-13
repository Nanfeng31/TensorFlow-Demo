# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:03:39 2020

@author: nanfeng
"""


import tensorflow as tf
from tensorflow import keras
import numpy as np

class Dataset:
    def __init__(self):
        self.fashion_mnist = keras.datasets.fashion_mnist        
        (train_images, self.train_labels), (test_images, self.test_labels) = self.fashion_mnist.load_data()
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0
        pass
    pass
