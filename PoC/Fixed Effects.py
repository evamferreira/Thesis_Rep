#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:01:17 2020

@author: evaferreira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df_list = [None]*15
df_test = [None]*15

for i in list(range(15)):
    df_list[i] = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/PoC/train_dfs/dftrain' + str(i) + '.csv')
    df_list[i].set_index(['PERMCO_Y'], inplace=True)

for i in list(range(15)):
    df_test[i] = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/PoC/test_dfs/dftest' + str(i) + '.csv')
    df_test[i].set_index(['PERMCO_Y'], inplace=True)
    
#%%

for i, col in enumerate(df_list[0].columns.to_list()):
    if col != 'y_t+2':
        df_list[0][col+'_mean'] = df_list[0][col].rolling(3).mean()
        df_list[0][col+'_mean'] = df_list[0].iloc[2::3, [17]]
        df_list[0][col+'_mean'].fillna(method='bfill', inplace=True)
        df_list[0][col] = df_list[0][col]-df_list[0][col+'_mean']
        df_list[0].drop(columns=[col+'_mean'], inplace=True)

#%%
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix

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

logreg = LogisticRegression()


#%%

logreg.fit(df_list[0].iloc[:,:16], df_list[0].iloc[:,16:])

#%%

predictions = logreg.predict(df_test[0].iloc[:,:16])

#%%
y_test = df_test[0].iloc[:,16:]['y_t+2'].values

#%%
f1 = f1_score(y_test, predictions)
acc = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
print(f1, acc, recall, precision, roc_auc)


#%%

predict_cm = confusion_matrix(y_test, predictions)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm, labels, title="Predictions \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

