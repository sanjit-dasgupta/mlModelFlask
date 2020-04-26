# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:21:27 2020

@author: hp
"""
import importlib
import tumorpred
importlib.reload(tumorpred)
model = tumorpred.TumorDetection()
model.load_image( r'images\user1\user_2.JPG')
result = model.predict_outcome()
if result[0] == 1:
    print("Tumor Detected")
else:
    print("Tumor Not Detected")
model.show_tumor()
