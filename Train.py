# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:51:23 2020

@author: Nanfeng
"""

import tensorflow as tf
from tensorflow import keras
from Model import Model
from Dataset import Dataset

class Train:
    def __init__(self):
        self.model = Model().model
        self.data = Dataset()
        self.model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            )
        self.test_images = self.data.test_images
        self.test_labels = self.data.test_labels
        self.train_images = self.data.train_images
        self.train_labels = self.data.train_labels
        pass
    
    def trainAim(self):
        self.model.fit(self.train_images, self.train_labels, epochs = 10)
    
    def predict(self):
        test_acc = self.model.evaluate(self.test_images, self.test_labels)
        predictions = self.model.predict(self.test_images)
        print("accuracy", test_acc)
        print("predictions:", predictions[0])
        print("true label is:", self.test_labels[0])
        pass
    """
    def main():

        pass
    
        
    if __name__ == "__main__":
        main()
        pass
    pass
    """
    
train = Train()
train.trainAim()
train.predict()


