'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''

from scripts.train import *

class TestModel :
    def __init__(self) :
        self.pred = None
    
    def test(self) :
        self.y_pred = TrainModel().model.predict(TrainModel().X_test)

