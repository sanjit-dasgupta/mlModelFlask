# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:07:49 2020

@author: hp
"""
# In[1]
# a) Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle

# In[2]
# b) Load dataset
dataframe = pd.read_csv("csv\dataset_processed.csv")
print(dataframe.head())

# In[3]
# 2. Summarize Data
# a) Descriptive statistics
print(dataframe.shape)
# Stats Display formatting
pd.set_option('display.width', 100) 
pd.set_option('precision', 3)
pd.set_option('display.max_columns', None) 
print(dataframe.describe())
print(dataframe.groupby('label').size())
correlation = dataframe.corr(method = "pearson") 
print(correlation) # -1 -> Negative, 0 -> No Corr, 1 -> Positive
skew = dataframe.skew()
print(skew)
# b) Data visualizations
# BoxPlot
dataframe.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()

dataframe.hist()
plt.show()

# In[4]
# 3. Prepare Data
X = dataframe.drop(["label"],axis=1)
y = dataframe["label"]

# In[5]
# a) Data Cleaning -> Alreadt clean
# c) Data Transforms
# Ratio Scaling
'''from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
#scaler = MinMaxScaler(feature_range=(0,1))
#scaler = StandardScaler()
scaler = Normalizer()
X_rescaled = scaler.fit_transform(X)'''

# 90% acc without Rescaling and 83 % with rescaling.

# b) Feature Selection
# SelectKbest with chi2
# In[6]
# Training Independent Var: X_rescaled, Dependent: y

# 4. Evaluate Algorithm.
#Splitting dataset into Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 7)
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(("RF",RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
'''
LR: 0.512334 (0.057842)
LDA: 0.523305 (0.052281)
KNN: 0.731901 (0.036263)
CART: 0.938749 (0.041401)
NB: 0.482320 (0.070638)
SVM: 0.515898 (0.050434)
RF: 0.936932 (0.039806)  # Highest

'''
# Build a Model

#model = RandomForestClassifier(n_estimators = 100, max_depth = 6) # 90% acc.
model = DecisionTreeClassifier() # 99% acc.: Makes No Sense.
model.fit(x_train,y_train) 
#from sklearn.svm import SVC
#model = SVC(kernel = 'rbf', random_state = 0)
#model.fit(x_train, y_train)

# In[]



# Make Predictions
# Predicting the Test set results
prediction = model.predict(x_test)

cm = confusion_matrix(y_test, prediction)
print(cm)
print(metrics.classification_report(prediction,y_test))
 
# In[]
with open('tree_v2.sav','wb') as file:
    pickle.dump(model, file)