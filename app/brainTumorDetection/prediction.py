# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:24:48 2020

@author: hp
"""
# In[]
import pandas as pd
import numpy as np
import joblib
import glob
import numpy as np
import cv2
import os
import csv
from PIL import Image
from os.path import dirname, join, abspath
from base64 import b64encode

# In[]
class TumorDetection():
    def __init__(self, model_file = 'tree_v2.sav'):
        model_file = join(dirname(abspath(__file__)), model_file)
        self.classifier = joblib.load(model_file)
        self.data = None
    def load_image(self,imgFileStorage):
        self.img = imgFileStorage.read()
        npimg = np.fromstring(self.img, np.uint8)
        self.img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        self.im = cv2.GaussianBlur(self.img,(5,5),2)
        self.im_gray = cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY)
        ret, self.thresh = cv2.threshold(self.im_gray,180,255,0)
        self.contours, _ = cv2.findContours(self.thresh,1,2)
        area = []
        for i in range(5):
            try:
                area.append(cv2.contourArea(self.contours[i]))
            except:
                area.append(0)
        self.data = np.array(area)
        self.data = np.reshape(self.data,(-1,5))
    def predict_outcome(self):
        pred = self.classifier.predict(self.data)
        return pred
    def show_tumor(self):
        im_save = cv2.drawContours(self.im,self.contours,-1,(0,255,0),3) 
        retval, buffer = cv2.imencode('.png', im_save)
        processed_image = b64encode(buffer).decode('ASCII')
        retval, buffer = cv2.imencode('.png', self.img)
        original_image = b64encode(buffer).decode('ASCII')
        return original_image, processed_image
    def predict(self, imgFileStorage):
        self.load_image(imgFileStorage)
        result = "Tumor Detected" if self.predict_outcome()[0] == 1 else "Tumor Not Detected"
        original_image, tumor_image = self.show_tumor()
        return (result, original_image, tumor_image)