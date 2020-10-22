#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:04:26 2020

@author: evaferreira
"""
# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


# Other Libraries
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
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

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)

#%%

from sklearn import svm
clf = svm.SVC()
clf.fit(Xsm_train, ysm_train)

#%%
oversample_bankruptcy_predictions = clf.predict(original_Xtest)

#%%
import itertools

# Create a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#%%

oversample_smote = confusion_matrix(original_ytest, oversample_bankruptcy_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%%

f1 = f1_score(original_ytest, oversample_bankruptcy_predictions)
acc = accuracy_score(original_ytest, oversample_bankruptcy_predictions)
recall = recall_score(original_ytest, oversample_bankruptcy_predictions)
precision = precision_score(original_ytest, oversample_bankruptcy_predictions)
print(f1, acc, recall, precision)

