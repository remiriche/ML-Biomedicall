# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:23:59 2023

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

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], linewidths=1, linestyles="dashed")
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, classification_report
Confusion_Matrix = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix, display_labels=logreg.classes_)
disp.plot()
plt.show()

print(classification_report(y_test, y_pred, target_names=['0','1']))
print("Erreur quadratique:", np.sqrt(mean_absolute_error(y_test, y_pred)))