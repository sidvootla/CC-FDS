# CC-FDS
Credit Card Fraud Detection System

pip install numpy
pip install pandas
pip install sklearn
pip install pandas_ml

import numpy as np
import pandas as pd
import sklearn

from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Creating a Data Frme with the dataset
df = pd.read_csv('creditcard.csv', low_memory=False)

# y = f(x) - > y: target ; f(): function/algorithm ; x: feature

df.head(5)

#Creating the Features, and Targets
x = df.iloc[::-1]
y = df['Class']

#Categorizing into Fraud and Not a Fraud

frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]

print("We have", len(frauds), "fraud data points and", len(non_frauds), "regular data points" )

# Scalarization of the Features to get combined a mean of 0 and variance of 0
x = scale(x)

#Splitting into Train and Test Data sets for the model building and Testing
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state=42 )

# Creating a Support Vector Machine Model
clf = svm.SVC()
clf.fit(X_train, y_train)

# Predicting the Model using Test dataset
preditctions = clf.predict(X_test)

# Getting the Confusion Matrix, Classification Report, and Accuracy Score
print(confusion_matrix(y_test, preditctions))
print(classification_report(y_test, preditctions))
print(accuracy_score(y_test, preditctions))
