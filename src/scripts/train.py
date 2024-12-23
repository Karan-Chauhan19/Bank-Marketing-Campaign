'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''
import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import sys
# sys.path.append('/home/karan-chauhan/WorkStation/Project/Bank-Marketing-Campaign/src')
from sklearn.model_selection import train_test_split
from nn_arch.architecture1 import Model
from scripts.preprocess_data import featureEngineering
from keras.callbacks import EarlyStopping
from config import config


class TrainModel :
    def __init__(self) :
        self.epochs = config.TRAINING["epochs"]
        self.batch_size = config.DATASET["batch_size"]
        self.history = None
        self.y_test = None
        self.X_test = None

    def train(self) :
        df = featureEngineering().get_clean_data()
        X_train,self.X_test,y_train,self.y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1],test_size=0.2,random_state=42)
        model = Model().build_model()

        callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,
        patience=20,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=False)

        self.history = model.fit(X_train,y_train,epochs=self.epochs,batch_size=self.batch_size,validation_data=(self.X_test,self.y_test),callbacks=callback)
        model.save('model.h5')

        return self.history
    
