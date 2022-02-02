# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:44:19 2021

@author: SelimPc
"""


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Import Data
train = pd.read_csv(r"./input/AI/prices.csv",dtype = np.float32)

# split data into features and price
targets_numpy = train.price.values
features_numpy = train.loc[:,train.columns != "price"].values 

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                              targets_numpy,
                                                                              test_size = 0.2,
                                                                              random_state = 42) 

#linear regression
reg = LinearRegression(normalize=False)
y_pred = reg.fit(features_train, targets_train).predict(features_test)
print("Mean squared error: %.2f" % mean_squared_error(targets_test, y_pred,squared=False))
test_accuracy = reg.score(features_test, targets_test)
print("test accuracy : %%%d"
      % (100*test_accuracy))

