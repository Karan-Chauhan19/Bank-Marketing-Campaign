'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''
# import sys
# import os
# sys.path.append(os.path.abspath('/home/karan-chauhan/WorkStation/Project/Bank-Marketing-Campaign'))


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from config import config

class Model :
    def __init__(self) :
        self.model = Sequential()
        self.optimizer = config.TRAINING["optimizer"]
        self.loss = config.TRAINING["loss_function"]
        self.activation = config.MODEL["activation"]
        self.batch_size = config.DATASET["batch_size"]
        self.epochs = config.TRAINING["epochs"]
        self.metrics = config.TRAINING["metrics"]
        self.layers = config.MODEL["layers"]
        self.input_dim = config.MODEL["input_dim"]

    def build_model(self) :
        c = 6
        self.model.add(Dense(128,activation=self.activation,input_dim=self.input_dim))
        for i in range(self.layers) :
            self.model.add(Dense((2)**c,activation=self.activation))
            self.model.add(Dropout())
            c -= 1
        self.model.add(Dense(1,activation='sigmoid'))
        
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metrics)
        return self.model