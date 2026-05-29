# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:37:55 2019

@author: Anubhav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
frame = pd.read_csv('data.csv')
frame = frame.iloc[:, :-1]

# Separating the dependent and independent variables
X = frame.iloc[:, 2:33].values
Y = frame.iloc[:, 1]

# Scaling the parameters
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encoding the independent variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# Splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Applying linear SVM
from sklearn.svm import SVC
linear_svm = SVC(kernel = 'linear', random_state = 0)
linear_svm.fit(X_train, Y_train)

"""__Prediction__"""
Y_pred = linear_svm.predict(X_test)
print(linear_svm.score(X_test, Y_test))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(Y_test, Y_pred)



