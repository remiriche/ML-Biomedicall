# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:12:43 2023

@author: matth
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = 'pca.user1.features_labels.csv'
df = pd.read_csv(dataset, delimiter=',')

from sklearn.model_selection import train_test_split
X = df.drop('active', axis=1).values
y = df['active'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn import svm
SVM = svm.SVC()
from sklearn.model_selection import GridSearchCV
param={'C': np.logspace(-1, 2, 20), 'gamma': np.logspace(-3, 1, 10)}
cv = GridSearchCV(SVM, param_grid=param)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

print(cv.best_params_)
print(cv.best_score_)

O_SVM = svm.SVC(C = cv.best_params_['C'], gamma = cv.best_params_['gamma'])
O_SVM.fit(X_train, y_train)
y_pred = O_SVM.predict(X_test)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = O_SVM.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], linewidths=1, linestyles="dashed")
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
Confusion_Matrix = confusion_matrix(y_test, y_pred, labels=O_SVM.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix, display_labels=O_SVM.classes_)
disp.plot()
plt.show()

from sklearn.metrics import recall_score, f1_score, accuracy_score, classification_report, hinge_loss
print(classification_report(y_test, y_pred, target_names=['0','1']))