# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:53:51 2023

@author: matth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import math

dataset = 'clean.user1.features_labels.csv'
df = pd.read_csv(dataset, delimiter=',')
X = df.drop('active', axis = 1).values
Y = df['active'].values
X = preprocessing.scale(X)

pca = PCA(n_components=2)
Z = pca.fit_transform(X)
pcadf = pd.DataFrame(data = np.append(Z, Y.reshape(-1,1), axis=1), columns=['pc1', 'pc2', 'active'])
pcadf.to_csv('pca.user1.features_labels.csv', index=False)
# fig_proj, ax = plt.subplots(figsize=(12, 10))
# ax.scatter(Z[:, 0], Z[:, 1],df['active'], c=(Y == 1))

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(Z[:, 0], Z[:, 1], df["active"], c=(Y == 1))
 
ax.set_xlabel('pc1')
ax.set_ylabel('pc2')
ax.set_zlabel('actif')
 
plt.show()

c1 =np.zeros(len(Z[:, 0]))
c2 =[]

# for i in Z[:, 0]:
#     y = 0.6*i-8.8
#     c1.append(y)
# for i in Z[:, 1]:
#     y = -0.27391487*i-8.8
#     c2.append(y)
# for i in range (len(Z[:, 0])):
#     y = 0.28444769*Z[i, 0]+0.02003898*Z[i, 1]-3.13866216
#     c1[i] = y

y = 0.28444778*Z[:, 0]-0.02003845*Z[:, 1]-3.13866383  
s = preprocessing.minmax_scale(y)
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_trisurf(Z[:, 0], Z[:, 1], y)
plt.show()
# ax.plot(Z[:, 0], Z[:, 1], c1 , c='blue')    
# ax.plot(Z[:, 1],c2 , c='red')

# fig_proj.show()