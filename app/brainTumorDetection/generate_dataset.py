# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:01:39 2020

@author: Arnob Chowdhury
"""
'''
Brain Tumour Detection Dataset preparation
1.open original image
2.convert to grayscale
3.Contour Detection
4.Get Areas of Largest Contours
--> For an Tumored Brian we may get many contours but for non tumoor brain 
we may get only one countour
'''

# TODO:
'''
1. Label 1 and 0
2. improve accuracy
'''
import glob
import numpy as np
import cv2
import os
import csv

label = "no" #yes for tumor images
dirlist = glob.glob(label+"\\*.jpg")
print(len(dirlist))
file = open("csv/dataset.csv","a")
for img_path in dirlist:
    im = cv2.imread(img_path,cv2.IMREAD_COLOR)
    im = cv2.GaussianBlur(im,(5,5),2)
    
    
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray,180,255,0) #for yes:180,  no:255. Is it OK?
    #gaus = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115,1)
    '''
    cv2.imshow(img_path,thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    contours, _ = cv2.findContours(thresh,1,2)
    #Detect Tumour.
    
    #cv2.drawContours(im,contours,-1,(0,255,0),3) 
    #cv2.imshow(img_path,thresh)
    #cv2.imshow(img_path,im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    outcome = '0'
    file.write(outcome)
    file.write(",")
    
    for i in range(5):
        try:
            area = cv2.contourArea(contours[i])
            file.write(str(area))
        except:
            file.write("0")
            
        file.write(",")
    file.write("\n")
    