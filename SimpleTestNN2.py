# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
print("X[0]",X[0])
print("y[0]", y[0])
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print(labels_train)
print("start fitting")
print(X_train[0,:])
print("X_train shape", X_train.shape[0], X_train.shape[1])
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test, predictions))

