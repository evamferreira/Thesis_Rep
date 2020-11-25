#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:51:14 2020

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

# LOGISTIC REGRESSION

    
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
plot_confusion_matrix(predict_cm, labels, title="Hazard \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%% 

#WEIGHTED CLASS LOGREG


#%%

w = {0:4, 1:96}

logreg_w = LogisticRegression(class_weight=w)


#%%

logreg_w.fit(df_list[0].iloc[:,:16], df_list[0].iloc[:,16:])

#%%

predictions_w = logreg_w.predict(df_test[0].iloc[:,:16])

#%%
y_test = df_test[0].iloc[:,16:]['y_t+2'].values

#%%
f1_w = f1_score(y_test, predictions_w)
acc_w = accuracy_score(y_test, predictions_w)
recall_w = recall_score(y_test, predictions_w)
precision_w = precision_score(y_test, predictions_w)
roc_auc_w = roc_auc_score(y_test, predictions_w)
print(f1_w, acc_w, recall_w, precision_w, roc_auc_w)

#%%

predict_cm_w = confusion_matrix(y_test, predictions_w)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_w, labels, title="Hazard w/ Class Weight \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%% 

#GRID SEARCH WEIGHTS


#%%
# define weight hyperparameter
gs_w = [{0:1000,1:100},{0:1000,1:10}, {0:1000,1:1.0}, 
     {0:500,1:1.0}, {0:400,1:1.0}, {0:300,1:1.0}, {0:200,1:1.0}, 
     {0:150,1:1.0}, {0:100,1:1.0}, {0:99,1:1.0}, {0:10,1:1.0}, 
     {0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100}, 
     {0:0.001,1:1.0}, {0:0.005,1:1.0}, {0:1.0,1:1.0}, 
     {0:1.0,1:0.1}, {0:10,1:0.1}, {0:100,1:0.1}, 
     {0:10,1:0.01}, {0:1.0,1:0.01}, {0:1.0,1:0.001}, {0:1.0,1:0.005}, 
     {0:1.0,1:10}, {0:1.0,1:99}, {0:1.0,1:100}, {0:1.0,1:150}, 
     {0:1.0,1:200}, {0:1.0,1:300},{0:1.0,1:400},{0:1.0,1:500}, 
     {0:1.0,1:1000}, {0:10,1:1000},{0:100,1:1000} ]
hyperparam_grid = {"class_weight": gs_w }

#%%
X = pd.concat([df_list[0],df_test[0]]).sort_index().iloc[:,:16]

Y = pd.concat([df_list[0],df_test[0]]).sort_index().iloc[:,16:]

for i in X.columns.to_list():    
    X[i] = rob_scaler.fit_transform(X[i].values.reshape(-1,1))
   
X = X.values
Y = Y.values.ravel()
#%%
from sklearn.model_selection import GridSearchCV
# define model
lg3 = LogisticRegression()

#%%
# define evaluation procedure
grid = GridSearchCV(lg3,hyperparam_grid,scoring="f1", cv=5)

#%%
grid.fit(X,Y)

#%%
print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

#%%

w = {0: 0.01, 1: 1.0}

logreg_w2 = LogisticRegression(class_weight=w)


#%%

logreg_w2.fit(df_list[0].iloc[:,:16], df_list[0].iloc[:,16:])

#%%

predictions_w2 = logreg_w2.predict(df_test[0].iloc[:,:16])

#%%
y_test = df_test[0].iloc[:,16:]['y_t+2'].values

#%%
f1_w2 = f1_score(y_test, predictions_w)
acc_w2 = accuracy_score(y_test, predictions_w)
recall_w2 = recall_score(y_test, predictions_w)
precision_w2 = precision_score(y_test, predictions_w)
roc_auc_w2 = roc_auc_score(y_test, predictions_w)
print(f1_w, acc_w, recall_w, precision_w, roc_auc_w)

#%%

predict_cm_w2 = confusion_matrix(y_test, predictions_w2)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_w2, labels, title="Hazard w/ Tuning \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%% 

# OVERSAMPLING


#%%
from imblearn.over_sampling import SMOTE
X_train = df_list[0].iloc[:,:16]
y_train = df_list[0].iloc[:,16:]

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

#%%

logreg_sm = LogisticRegression()


#%%

logreg_sm.fit(Xsm_train, ysm_train)

#%%

predictions_sm = logreg_sm.predict(df_test[0].iloc[:,:16])

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
plot_confusion_matrix(predict_cm_sm, labels, title="Hazard w/ Oversampling \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
    
    