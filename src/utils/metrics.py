'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''

from sklearn.metrics import classification_report 
from scripts.test import *

class metrics :
    def __init__(self) :
        pass

    def accuracy(self) :

        y_pred = (TestModel().y_pred>0.5).astype(int)

        print(classification_report(TrainModel().y_test,y_pred))