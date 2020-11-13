#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:50:14 2020

@author: evaferreira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix


df_list = [None]*15
df_test = [None]*15

for i in list(range(15)):
    df_list[i] = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/PoC/train_dfs/dftrain' + str(i) + '.csv')
    df_list[i].set_index(['PERMCO_Y'], inplace=True)

for i in list(range(15)):
    df_test[i] = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/PoC/test_dfs/dftest' + str(i) + '.csv')
    df_test[i].set_index(['PERMCO_Y'], inplace=True)

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

# Rescale the data
from sklearn.preprocessing import RobustScaler

# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for i in df_list[0].iloc[:,:16].columns.to_list():    
    df_list[0].iloc[:,:16][i] = rob_scaler.fit_transform(df_list[0].iloc[:,:16][i].values.reshape(-1,1))
    
for i in df_test[0].iloc[:,:16].columns.to_list():    
    df_test[0].iloc[:,:16][i] = rob_scaler.fit_transform(df_test[0].iloc[:,:16][i].values.reshape(-1,1))
    
#%%

from sklearn import svm
clf = svm.SVC()
clf.fit(df_list[0].iloc[:,:16], df_list[0].iloc[:,16:])

#%%
predictions = clf.predict(df_test[0].iloc[:,:16])

#%%
y_test = df_test[0].iloc[:,16:]['y_t+2'].values

#%%

predict_cm = confusion_matrix(y_test, predictions)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm, labels, title="SVM Predictions \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%%

f1 = f1_score(y_test, predictions)
acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
print(f1, acc, recall, precision, roc_auc)

#%% 

# OVERSAMPLING


#%%
X_train = df_list[0].iloc[:,:16]
y_train = df_list[0].iloc[:,16:]

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

#%%

clf_sm = svm.SVC()
clf_sm.fit(df_list[0].iloc[:,:16], df_list[0].iloc[:,16:])

#%%
predictions_sm = clf_sm.predict(df_test[0].iloc[:,:16])

#%%
y_test = df_test[0].iloc[:,16:]['y_t+2'].values

#%%
f1_sm = f1_score(y_test, predictions_sm)
acc_sm = accuracy_score(y_test, predictions_sm)
recall_sm = recall_score(y_test, predictions_sm)
precision_sm = precision_score(y_test, predictions_sm)
roc_auc_sm = roc_auc_score(y_test, predictions_sm)
print(f1_sm, acc_sm, recall_sm, precision_sm, roc_auc_sm)


#%%

predict_cm_sm = confusion_matrix(y_test, predictions_sm)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_sm, labels, title="SVM Oversample Predictions \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
    

#%%

# GRID SEARCH FOR TUNING PARAMETERS


#%%
from sklearn.model_selection import GridSearchCV

# define model
svm_gs = svm.SVC()

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'sigmoid'], 
              'class_weight': [{0:1, 1:1,}, {0:2,1:98}, {0:1, 1:100}]}

grid = GridSearchCV(svm.SVC(),param_grid, scoring="f1", verbose=2)
grid.fit(X_train,y_train)

#%%

print(grid.best_params_)

#%%
svm_gs = svm.SVC(C=1, gamma=0.01, kernel='rbf', class_weight={0:2, 1:98})
svm_gs.fit(df_list[0].iloc[:,:16], df_list[0].iloc[:,16:])

#%%
predictions_gs = svm_gs.predict(df_test[0].iloc[:,:16])

#%%
y_test = df_test[0].iloc[:,16:]['y_t+2'].values

#%%

f1_sm = f1_score(y_test, predictions_gs)
acc_sm = accuracy_score(y_test, predictions_gs)
recall_sm = recall_score(y_test, predictions_gs)
precision_sm = precision_score(y_test, predictions_gs)
roc_auc_sm = roc_auc_score(y_test, predictions_gs)
print(f1_sm, acc_sm, recall_sm, precision_sm, roc_auc_sm)


#%%

predict_cm_gs = confusion_matrix(y_test, predictions_gs)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_gs, labels, title="SVM Oversample Predictions \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
    

