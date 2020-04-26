# -*- coding: utf-8 -*-
"""
Created on Tue March 20 15:21:48 2020
Diabetes Prediciton
@author: Arnob Chowdhury

Changes: Removed analysis codes
System Integration file is of no use
Applied pipeline.(Can be improved)
"""
# In[]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib, pickle

# In[]
#from sklearn.pipeline import Pipeline
# Stats Display formatting
pd.set_option('display.width', 100) 
pd.set_option('precision', 3)
pd.set_option('display.max_columns', None) 
# Findiing Stats of the data
dataframe = pd.read_csv("diabetes.csv")
print(dataframe.head(20))

# In[]
shape = dataframe.shape
print(shape)
types = dataframe.dtypes
print(types)

description = dataframe.describe()  # All stats here!
print(description) 

class_cnt = dataframe.groupby('Outcome').size()
print(class_cnt)

'''Graph needed'''
correlation = dataframe.corr(method = "pearson") 
print(correlation) # -1 -> Negative, 0 -> No Corr, 1 -> Positive
skew = dataframe.skew()
print(skew)

# In[]

# Visualization

# 1. Histogram
dataframe.hist()
plt.show()

#2. Density Graph
dataframe.plot(kind="density",subplots= True, layout=(3,3), sharex = False)
plt.show()

# 3. Correlation Plot (color graph)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

# In[]
# Pre processing the data
X = dataframe.drop(["Outcome"],axis =1)
y = dataframe["Outcome"]

# In[]

# 1. Best Feature Selection
# Univariate Selection
from sklearn.feature_selection import SelectKBest, chi2
test = SelectKBest(score_func = chi2,k=4) # Chose best 4 features
fit = test.fit(X,y)
print(fit.scores_)
X = fit.transform(X)
print(X[0:5,:]) # 77% accuracy
# In[]

# Model Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Model Evaluations with Kfolds
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # acc:76 %
# In[]
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(),LogisticRegression())
# In[]
pipe.fit(X_train, y_train)
# In[]
y_pred = pipe.predict(X_test)
# In[]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn import metrics
print(metrics.classification_report(y_pred,y_test)) # 77.0%
# In[]
with open('modelv3.sav','wb') as file:
    pickle.dump(pipe, file)