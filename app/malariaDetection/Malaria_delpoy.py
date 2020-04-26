# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 00:31:05 2020

@author: hp
"""

import malariadetect
import importlib
importlib.reload(malariadetect)
model = malariadetect.MalariaDetection()
model.load_image( r'images\user_1\user_1.png')
#print(model.data)
result = model.predict_outcome()
if result[0] == "Parasitized":
    print("Plasmodium Detected")
else:
    print("Plasmodium Not Detected")
model.show_plasmodium()