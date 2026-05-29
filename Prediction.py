# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 00:57:02 2019

@author: Anubhav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Wisconsin Breast Cancer Dataset
frame = pd.read_csv('data.csv')

# Segregating into dependent and independent variables
X = frame.iloc[:,2:32].values
Y = frame.iloc[:, 1:2]
frame = pd. DataFrame(X)


# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
Y = encode.fit_transform(Y)

# Feature scaling the dataset using MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Splitting into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

# Backward Elimiantion based on significance level and p-value
import statsmodels.formula.api as sts


classifier_ols = sts.OLS(endog = Y, exog = X).fit()
classifier_ols.summary()
pvals = (classifier_ols.pvalues).astype(float)
pvals = pvals.reshape(30,1)

import seaborn as sns
sns.set(style= 'ticks', color_codes=True)
plt.figure(figsize=(14, 12))
sns.heatmap(frame.astype(float).corr(), linewidths=0.1, square=True, linecolor='white', annot=True)
plt.show()














