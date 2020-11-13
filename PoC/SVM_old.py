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
    #print("Train:", train_index, "Test:", test_index)
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

#%%


#CS SVM


#%%
X_CSdf = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/X_CS.csv', index_col = 'PERMCO')
Y_CSdf = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/Y_CS.csv', index_col = 'PERMCO')
X_CSdf.head()
Y_CSdf.head()

#%%

# Rescale the data
# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for i in X_CSdf.columns.to_list():    
    X_CSdf[i] = rob_scaler.fit_transform(X_CSdf[i].values.reshape(-1,1))

X_CSdf.head()

#%%
colors = ["#0101DF", "#DF0101"]

sns.countplot('dlrsn', data=Y_CSdf, palette=colors)
plt.title('Bankruptcy Distributions \n (0: No-Bankrupt || 1: Bankrupt)', fontsize=14)

#%%

print('Non-Bankrupt', round(Y_CSdf['dlrsn'].value_counts()[0]/len(Y_CSdf) * 100,2), '% of the dataset')
print('Bankrupt', round(Y_CSdf['dlrsn'].value_counts()[1]/len(Y_CSdf) * 100,2), '% of the dataset')

X_CS = X_CSdf
y_CS = Y_CSdf['dlrsn']

sss = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

for train_index_CS, test_index_CS in sss.split(X_CS, y_CS):
    #print("Train:", train_index_CS, "Test:", test_index_CS)
    original_Xtrain_CS, original_Xtest_CS = X_CS.iloc[train_index_CS], X_CS.iloc[test_index_CS]
    original_ytrain_CS, original_ytest_CS = y_CS.iloc[train_index_CS], y_CS.iloc[test_index_CS]

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain_CS = original_Xtrain_CS.values
original_Xtest_CS = original_Xtest_CS.values
original_ytrain_CS = original_ytrain_CS.values
original_ytest_CS = original_ytest_CS.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain_CS, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest_CS, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain_CS))
print(test_counts_label/ len(original_ytest_CS))


#%%

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train_CS, ysm_train_CS = sm.fit_sample(original_Xtrain_CS, original_ytrain_CS)

#%%

from sklearn import svm
clf = svm.SVC()
clf.fit(Xsm_train_CS, ysm_train_CS)

#%%
oversample_bankruptcy_predictions_CS = clf.predict(original_Xtest_CS)

#%%

oversample_smote_CS = confusion_matrix(original_ytest_CS, oversample_bankruptcy_predictions_CS)
actual_cm_CS = confusion_matrix(original_ytest_CS, original_ytest_CS)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote_CS, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm_CS, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%%

f1_CS = f1_score(original_ytest_CS, oversample_bankruptcy_predictions_CS)
acc_CS = accuracy_score(original_ytest_CS, oversample_bankruptcy_predictions_CS)
recall_CS = recall_score(original_ytest_CS, oversample_bankruptcy_predictions_CS)
precision_CS = precision_score(original_ytest_CS, oversample_bankruptcy_predictions_CS)
print(f1_CS, acc_CS, recall_CS, precision_CS)

#%%


#IT SVM


#%%
X_ITdf = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/X_IT.csv', index_col = 'PERMCO')
Y_ITdf = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/Y_IT.csv', index_col = 'PERMCO')
X_ITdf.head()
Y_ITdf.head()

#%%

# Rescale the data
# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for i in X_ITdf.columns.to_list():    
    X_ITdf[i] = rob_scaler.fit_transform(X_ITdf[i].values.reshape(-1,1))

X_ITdf.head()

#%%
colors = ["#0101DF", "#DF0101"]

sns.countplot('dlrsn', data=Y_ITdf, palette=colors)
plt.title('Bankruptcy Distributions \n (0: No-Bankrupt || 1: Bankrupt)', fontsize=14)

#%%

print('Non-Bankrupt', round(Y_ITdf['dlrsn'].value_counts()[0]/len(Y_ITdf) * 100,2), '% of the dataset')
print('Bankrupt', round(Y_ITdf['dlrsn'].value_counts()[1]/len(Y_ITdf) * 100,2), '% of the dataset')

X_IT = X_ITdf
y_IT = Y_ITdf['dlrsn']

sss = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

for train_index_IT, test_index_IT in sss.split(X_IT, y_IT):
    #print("Train:", train_index_IT, "Test:", test_index_IT)
    original_Xtrain_IT, original_Xtest_IT = X_IT.iloc[train_index_IT], X_IT.iloc[test_index_IT]
    original_ytrain_IT, original_ytest_IT = y_IT.iloc[train_index_IT], y_IT.iloc[test_index_IT]

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain_IT = original_Xtrain_IT.values
original_Xtest_IT = original_Xtest_IT.values
original_ytrain_IT = original_ytrain_IT.values
original_ytest_IT = original_ytest_IT.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain_IT, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest_IT, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain_IT))
print(test_counts_label/ len(original_ytest_IT))


#%%

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train_IT, ysm_train_IT = sm.fit_sample(original_Xtrain_IT, original_ytrain_IT)

#%%

from sklearn import svm
clf = svm.SVC()
clf.fit(Xsm_train_IT, ysm_train_IT)

#%%
oversample_bankruptcy_predictions_IT = clf.predict(original_Xtest_IT)

#%%

oversample_smote_IT = confusion_matrix(original_ytest_IT, oversample_bankruptcy_predictions_IT)
actual_cm_IT = confusion_matrix(original_ytest_IT, original_ytest_IT)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote_IT, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm_IT, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%%

f1_IT = f1_score(original_ytest_IT, oversample_bankruptcy_predictions_IT)
acc_IT = accuracy_score(original_ytest_IT, oversample_bankruptcy_predictions_IT)
recall_IT = recall_score(original_ytest_IT, oversample_bankruptcy_predictions_IT)
precision_IT = precision_score(original_ytest_IT, oversample_bankruptcy_predictions_IT)
print(f1_IT, acc_IT, recall_IT, precision_IT)

#%%


#FS SVM


#%%
X_FSdf = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/X_FS.csv', index_col = 'PERMCO')
Y_FSdf = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/Y_FS.csv', index_col = 'PERMCO')
X_FSdf.head()
Y_FSdf.head()

#%%

# Rescale the data
# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for i in X_FSdf.columns.to_list():    
    X_FSdf[i] = rob_scaler.fit_transform(X_FSdf[i].values.reshape(-1,1))

X_FSdf.head()

#%%
colors = ["#0101DF", "#DF0101"]

sns.countplot('dlrsn', data=Y_FSdf, palette=colors)
plt.title('Bankruptcy Distributions \n (0: No-Bankrupt || 1: Bankrupt)', fontsize=14)

#%%

print('Non-Bankrupt', round(Y_FSdf['dlrsn'].value_counts()[0]/len(Y_FSdf) * 100,2), '% of the dataset')
print('Bankrupt', round(Y_FSdf['dlrsn'].value_counts()[1]/len(Y_FSdf) * 100,2), '% of the dataset')

X_FS = X_FSdf
y_FS = Y_FSdf['dlrsn']

sss = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

for train_index_FS, test_index_FS in sss.split(X_FS, y_FS):
    #print("Train:", train_index_FS, "Test:", test_index_FS)
    original_Xtrain_FS, original_Xtest_FS = X_FS.iloc[train_index_FS], X_FS.iloc[test_index_FS]
    original_ytrain_FS, original_ytest_FS = y_FS.iloc[train_index_FS], y_FS.iloc[test_index_FS]

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain_FS = original_Xtrain_FS.values
original_Xtest_FS = original_Xtest_FS.values
original_ytrain_FS = original_ytrain_FS.values
original_ytest_FS = original_ytest_FS.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain_FS, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest_FS, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain_FS))
print(test_counts_label/ len(original_ytest_FS))


#%%

# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


# This will be the data were we are going to 
Xsm_train_FS, ysm_train_FS = sm.fit_sample(original_Xtrain_FS, original_ytrain_FS)

#%%

from sklearn import svm
clf = svm.SVC()
clf.fit(Xsm_train_FS, ysm_train_FS)

#%%
oversample_bankruptcy_predictions_FS = clf.predict(original_Xtest_FS)

#%%

oversample_smote_FS = confusion_matrix(original_ytest_FS, oversample_bankruptcy_predictions_FS)
actual_cm_FS = confusion_matrix(original_ytest_FS, original_ytest_FS)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(oversample_smote_FS, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm_FS, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%%

f1_FS = f1_score(original_ytest_FS, oversample_bankruptcy_predictions_FS)
acc_FS = accuracy_score(original_ytest_FS, oversample_bankruptcy_predictions_FS)
recall_FS = recall_score(original_ytest_FS, oversample_bankruptcy_predictions_FS)
precision_FS = precision_score(original_ytest_FS, oversample_bankruptcy_predictions_FS)
print(f1_FS, acc_FS, recall_FS, precision_FS)


