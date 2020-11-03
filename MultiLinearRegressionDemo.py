# -*- coding: utf-8 -*-

from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r"/home/nyaruko/ml/delivery1.csv"
deliveryData = genfromtxt(dataPath, delimiter=',')

print("data:", deliveryData)

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]
# X[0][0]=100
print('X:', X, 'Y', Y)

regr = linear_model.LinearRegression()

regr.fit(X,Y)
print('coefficients', regr.coef_)
print('intercept',regr.intercept_)

xP =  [[102,6]]
yP = regr.predict(xP)
print(yP)
