#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:01:57 2020

@author: evaferreira
"""
# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import time

# Classifier Libraries
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


X_df = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/X.csv', index_col = 'PERMCO')
Y_df = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/Y.csv', index_col = 'PERMCO')
X_df.head()
Y_df.head()

#%%

# Rescale the data
from sklearn.preprocessing import RobustScaler

# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for i in X_df.columns.to_list():    
    X_df[i] = rob_scaler.fit_transform(X_df[i].values.reshape(-1,1))

X_df.head()

#%%
colors = ["#0101DF", "#DF0101"]

sns.countplot('dlrsn', data=Y_df, palette=colors)
plt.title('Bankruptcy Distributions \n (0: No-Bankrupt || 1: Bankrupt)', fontsize=14)

#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print('Non-Bankrupt', round(Y_df['dlrsn'].value_counts()[0]/len(Y_df) * 100,2), '% of the dataset')
print('Bankrupt', round(Y_df['dlrsn'].value_counts()[1]/len(Y_df) * 100,2), '% of the dataset')

X = X_df
y = Y_df['dlrsn']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


#%%
