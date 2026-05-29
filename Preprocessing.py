# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 02:21:32 2019

@author: Anubhav
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
frame = pd.read_csv('data.csv')
dataset = frame.iloc[:, 2:32]

# Outlier detection and removal using Clamp Transformation
for i in range(0, 28, 1):
    q1 = dataset.iloc[:, i].quantile(0.25)
    q2 = dataset.iloc[:, i].quantile(0.75)
    quart_range = q2 - q1
    for j in range(0, 569):
        if dataset.iloc[j][i] < (q1 - 1.5 * quart_range):
            dataset.iloc[j][i] = q1 - 1.5 * quart_range
        if dataset.iloc[j][i] > (q2 + 1.5 * quart_range):
            dataset.iloc[j][i] = q2 + 1.5 * quart_range

# Segregating into dependent and independent variables
X = dataset.values
Y = frame.iloc[:, 1]

# Encoding the independent variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
Y = Y.reshape(569,1)



# Feature Selection using Backward Elimination
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        classifier_OLS = sm.OLS(Y, x).fit()
        maxVar = max(classifier_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (classifier_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    classifier_OLS.summary()
    return x

X_opt = backwardElimination(X, 0.05)

# Scaling the parameters
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_opt = scaler.fit_transform(X_opt)

# Splitting into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_opt, Y, test_size = 0.3, random_state = 0)


# Classsification based on Linear SVM
from sklearn.svm import SVC
kernel_svm = SVC(kernel = 'linear', random_state = 0)
kernel_svm.fit(X_train, Y_train)

Y_pred = kernel_svm.predict(X_test)
print(kernel_svm.score(X_test, Y_test))


# Classsification based on Kernel SVM
from sklearn.svm import SVC
kernel_svm = SVC(kernel = 'rbf', random_state = 0)
kernel_svm.fit(X_train, Y_train)

Y_pred = kernel_svm.predict(X_test)
print(kernel_svm.score(X_test, Y_test))

from sklearn.model_selection import cross_val_score
validator = cross_val_score(estimator = kernel_svm, X = X_train, y = Y_train, cv = 10)
validator.mean()

# Classification based on Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
            
Y_NB = classifier.predict(X_test)



# Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)

forest_pred = forest.predict(X_test)
print(forest.score(X_test, Y_test))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(Y_test, forest_pred)


# Classification using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p = 2)
knn.fit(X_train, Y_train)

knn_pred = knn.predict(X_test)
knn_pred = knn_pred.reshape(171,1)

knn.score(X_test, Y_test)





features_mean = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean']
data_drop = frame.drop('diagnosis',axis=1)
data_drop = data_drop[features_mean]
for index,columns in enumerate(data_drop):
    plt.figure(index)
    plt.figure(figsize=(5,5))
    sns.stripplot(x='diagnosis', y= 'symmetry_mean', data= frame, jitter=True, palette = 'Set1');
    sns.plt.title('Diagnosis vs symmetry mean')


plt.figure(figsize = (20,20))
plt.scatter(Y_test, Y_NB, color = 'blue')
plt.show()














