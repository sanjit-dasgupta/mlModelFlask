# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:31:55 2020

@author: hp
"""
import importlib
import prediction
import numpy as np
dataframe_instance = []
msg = ["Enter Glucose reading","Enter Insulin reading","Enter BMI","Enter age"]
for i in range(4):
    read = float(input())
    dataframe_instance.append(read)
data_np = np.array(dataframe_instance)
data_np = np.reshape(data_np,(-1,4))    
model = prediction.DiabetesModel('modelv3.sav')
print(model.predict_outcome(data_np))
print(model.predict_proba(data_np))
