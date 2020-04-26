# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:15:21 2020

@author: Arnob
"""
# In[]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import pickle
import numpy as np
# In[]
dataframe = pd.read_csv("csv\dataset_processed.csv")
print(dataframe.head())

#Splitting dataset into Training set and Test set

x = dataframe.drop(["Label"],axis=1)
y = dataframe["Label"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 4)

# Build a Model

model = RandomForestClassifier(n_estimators = 100, max_depth = 6,criterion="entropy")
#model = DecisionTreeClassifier()
model.fit(x_train,y_train)

# Make Predictions
# In[]
area = [0,0,135.5,0,0]
area = np.array(area)
area = np.reshape(area,(-1,5))
prede = model.predict(area)
#print(prede, y_test)
# In[]
#print(metrics.classification_report(prede,y_test))
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, prediction)
#print(cm)###

# In[]
with open('rf_ensemblev3.sav','wb') as file:
    pickle.dump(model, file)
