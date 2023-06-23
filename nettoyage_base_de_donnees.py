# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:43:06 2023

@author: matth
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

df = pd.read_csv(r'C:\Users\remir\Desktop\Projet ML\user1.features_labels.csv', delimiter=',')

#Donne le nombre des cases NaN pour chacune des colonnes
print(df.isnull().sum())
print(df.info())


dfn1 = df.dropna(axis=1, how='all', inplace=False) # On retire les colonnes entierement vide
# cdf = df.dropna(axis=1, thresh=1913, inplace=False) # seuil à 1913 ?
target = dfn1.iloc[:,219:]
u=target.T.isnull().sum()

expreg = re.compile('^label[:_]?.*')
numerical_variables = []
for col in dfn1.columns:
    if not expreg.match(col):
        numerical_variables.append(col) # Liste des features
        
lbex = dfn1[['label:SLEEPING','label:SITTING','label:LYING_DOWN','label:LOC_home','label:FIX_running','label:STAIRS_-_GOING_UP','label:FIX_walking']].values
act=np.zeros(2685)
for i in range (len(lbex)):
    for j in range (len(lbex.T)):
        if lbex[i,j]==1 and j>3:
            act[i]=1
    if sum(lbex[i])!=1 :
        act[i]='nan'
        

X = df[numerical_variables].values
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X=imp_mean.fit_transform(X)

active_df=np.append(X, act.reshape(-1,1), axis=1)
#création d'une dataframe normaliser depuis un tableau
df_names=numerical_variables
df_names.append('active')
dfn2 = pd.DataFrame(active_df,columns=[df_names])
clean_df = dfn2.dropna(axis=0, how='any')

clean_df.to_csv('clean.user1.features_labels.csv', index=False)

Xn = preprocessing.normalize(X) #normalisation

active_df=np.append(Xn, act.reshape(-1,1), axis=1)
#création d'une dataframe normaliser depuis un tableau
dfn2 = pd.DataFrame(active_df,columns=[df_names])
clean_df_n = dfn2.dropna(axis=0, how='any')

clean_df_n.to_csv('clean.user1.features_labels.n.csv', index=False)


Xs = preprocessing.scale(X) #Standardisation

active_df=np.append(Xs, act.reshape(-1,1), axis=1)
#création d'une dataframe standardiser depuis un tableau
dfn2 = pd.DataFrame(active_df,columns=[df_names])
clean_df_s = dfn2.dropna(axis=0, how='any')

clean_df_s.to_csv('clean.user1.features_labels.s.csv', index=False)