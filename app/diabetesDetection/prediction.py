# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:31:57 2020

@author: Arnob
"""
# Creating a Module

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib, pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from os.path import dirname, join, abspath

class DiabetesModel():
    def __init__(self,model_file='modelv3.sav'):
        model_file = join(dirname(abspath(__file__)), model_file)
        with open(model_file,'rb') as model_file:
            self.classifier  = pickle.load(model_file)
            self.data = None
    # reading is an array to be passed from GUI
    def predict_outcome(self,reading):
        self.data = reading
        if self.data is not None:
            pred = self.classifier.predict(self.data)
            return pred
        else:
            return None
    def predict_proba(self,reading):
        self.data = reading
        if self.data is not None:
            prob = self.classifier.predict_proba(self.data)[:,1]
            return prob
        else:
            return None
    def predict(self, glucose, insulin, bmi, age):
        dataframe_instance = list(map(float, [glucose, insulin, bmi, age]))
        data_np = np.array(dataframe_instance)
        data_np = np.reshape(data_np,(-1,4))
        outcome = self.predict_outcome(data_np)
        probability = self.predict_proba(data_np)
        return tuple((outcome, probability))