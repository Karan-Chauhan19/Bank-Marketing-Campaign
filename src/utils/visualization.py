'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
from src.scripts.train import *

class Visualization :
    def __init__(self) :
        pass

    def plot_loss(self):
        plt.plot(TrainModel().history.history['loss'],label='loss')
        plt.plot(TrainModel().history.history['val_loss'],label='val_loss')
        plt.legend()
        plt.savefig(os.path.join('/home/karan-chauhan/WorkStation/Project/Bank-Marketing-Campaign/results','loss-vs-val_loss'))
        # plt.show()
    
    def plot_accuracy(self):
        plt.plot(TrainModel().history.history['accuracy'],label='accuracy')
        plt.plot(TrainModel().history.history['val_accuracy'],label='val_accuracy')
        plt.legend()
        plt.savefig(os.path.join('/home/karan-chauhan/WorkStation/Project/Bank-Marketing-Campaign/results','accuracy-vs-val_accuracy'))
        # plt.show()

