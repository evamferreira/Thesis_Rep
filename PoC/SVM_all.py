#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:24:50 2020

@author: evaferreira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm

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

for df_ in list(range(len(df_list))):
    for i in df_list[df_].iloc[:,:16].columns.to_list():    
        df_list[df_].iloc[:,:16][i] = rob_scaler.fit_transform(df_list[df_].iloc[:,:16][i].values.reshape(-1,1))  
        df_test[df_].iloc[:,:16][i] = rob_scaler.fit_transform(df_test[df_].iloc[:,:16][i].values.reshape(-1,1))

#%%

def svc_nosm_notun(df, df_t):
    
    svc = svm.SVC(kernel='rbf')
    svc.fit(df.iloc[:,:16], df.iloc[:,16:])

    predictions = svc.predict(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions = [[] for i in range(15)]
y_tests = [[] for i in range(15)]
f1s = [None for i in range(15)]
accs = [None for i in range(15)]
recalls = [None for i in range(15)]
precisions = [None for i in range(15)]
for i in list(range(15)):
    predictions[i], y_tests[i], f1s[i], accs[i], recalls[i], precisions[i] = svc_nosm_notun(df_list[i], df_test[i])

#%%

for i in list(range(15)):
    y_tests[i] = y_tests[i].reshape(y_tests[i].shape[0],1)
    predictions[i] = predictions[i].reshape(predictions[i].shape[0],1)

#%%
predict_cm = confusion_matrix(np.vstack(y_tests), np.vstack(predictions))
actual_cm = confusion_matrix(np.vstack(y_tests), np.vstack(y_tests))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm, labels, title="SVM w/o Oversampling w/o Tuning \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('Basic (f1, acc, recall, precision):', np.mean(f1s), np.mean(accs), np.mean(recalls), np.mean(precisions))

#%%

# TUNING 

#%%

def svc_nosm_tun(df, df_t):
    
    svc = svm.SVC(C=1, gamma=0.01, kernel='rbf', class_weight={0:4, 1:96})
    svc.fit(df.iloc[:,:16], df.iloc[:,16:])

    predictions = svc.predict(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_tn = [[] for i in range(15)]
y_tests_tn = [[] for i in range(15)]
f1s_tn = [None for i in range(15)]
accs_tn = [None for i in range(15)]
recalls_tn = [None for i in range(15)]
precisions_tn = [None for i in range(15)]
for i in list(range(15)):
    predictions_tn[i], y_tests_tn[i], f1s_tn[i], accs_tn[i], recalls_tn[i], precisions_tn[i] = svc_nosm_tun(df_list[i], df_test[i])

#%%

for i in list(range(15)):
    y_tests_tn[i] = y_tests_tn[i].reshape(y_tests_tn[i].shape[0],1)
    predictions_tn[i] = predictions_tn[i].reshape(predictions_tn[i].shape[0],1)

#%%
predict_cm_tn = confusion_matrix(np.vstack(y_tests_tn), np.vstack(predictions_tn))
actual_cm = confusion_matrix(np.vstack(y_tests_tn), np.vstack(y_tests_tn))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_tn, labels, title="SVM w/o Oversampling w/ Tuning \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('TUN (f1, acc, recall, precision):', np.mean(f1s_tn), np.mean(accs_tn), np.mean(recalls_tn), np.mean(precisions_tn))


#%%

# OVERSAMPLING

#%%

def svc_sm(df, df_t):
    
    X_train = df.iloc[:,:16]
    y_train = df.iloc[:,16:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    svc = svm.SVC(kernel='rbf')
    svc.fit(Xsm_train, ysm_train)

    predictions = svc.predict(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_sm = [[] for i in range(15)]
y_tests_sm = [[] for i in range(15)]
f1s_sm = [None for i in range(15)]
accs_sm = [None for i in range(15)]
recalls_sm = [None for i in range(15)]
precisions_sm = [None for i in range(15)]

for i in list(range(15)):
    predictions_sm[i], y_tests_sm[i], f1s_sm[i], accs_sm[i], recalls_sm[i], precisions_sm[i] = svc_sm(df_list[i], df_test[i])

#%%

for i in list(range(15)):
    y_tests_sm[i] = y_tests_sm[i].reshape(y_tests_sm[i].shape[0],1)
    predictions_sm[i] = predictions_sm[i].reshape(predictions_sm[i].shape[0],1)
    
#%%
predict_cm_sm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(predictions_sm))
actual_cm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(y_tests_sm))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_sm, labels, title="SVM w/ Oversampling w/o Tuning \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SM (f1, acc, recall, precision):', np.mean(f1s_sm), np.mean(accs_sm), np.mean(recalls_sm), np.mean(precisions_sm))

#%% 

# INDUSTRIES


#%%
df_ind = [None]*15
df_tind = [None]*15

for i in list(range(15)):
    df_ind[i] = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/PoC/train_dfs_ind/dftrain' + str(i) + 'ind.csv')
    df_ind[i].set_index(['PERMCO_Y'], inplace=True)

for i in list(range(15)):
    df_tind[i] = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/PoC/test_dfs_ind/dftest' + str(i) + 'ind.csv')
    df_tind[i].set_index(['PERMCO_Y'], inplace=True)
    

#%%

# Rescale the data
from sklearn.preprocessing import RobustScaler

# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for df_ in list(range(len(df_list))):
    for i in df_ind[df_].iloc[:,:16].columns.to_list():    
        df_ind[df_].iloc[:,:16][i] = rob_scaler.fit_transform(df_ind[df_].iloc[:,:16][i].values.reshape(-1,1))  
        df_tind[df_].iloc[:,:16][i] = rob_scaler.fit_transform(df_tind[df_].iloc[:,:16][i].values.reshape(-1,1))

#%%

def svc_nosm_notun_ind(df, df_t):
    
    svc = svm.SVC(kernel='rbf')
    svc.fit(df.iloc[:,:26], df.iloc[:,26:])

    predictions = svc.predict(df_t.iloc[:,:26])

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_i = [[] for i in range(15)]
y_tests_i = [[] for i in range(15)]
f1s_i = [None for i in range(15)]
accs_i = [None for i in range(15)]
recalls_i = [None for i in range(15)]
precisions_i = [None for i in range(15)]
for i in list(range(15)):
    predictions_i[i], y_tests_i[i], f1s_i[i], accs_i[i], recalls_i[i], precisions_i[i] = svc_nosm_notun_ind(df_ind[i], df_tind[i])

#%%

for i in list(range(15)):
    y_tests_i[i] = y_tests_i[i].reshape(y_tests_i[i].shape[0],1)
    predictions_i[i] = predictions_i[i].reshape(predictions_i[i].shape[0],1)

#%%
predict_cmi = confusion_matrix(np.vstack(y_tests_i), np.vstack(predictions_i))
actual_cm = confusion_matrix(np.vstack(y_tests_i), np.vstack(y_tests_i))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cmi, labels, title="SVM w/o Oversampling w/o Tuning (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('I (f1, acc, recall, precision):', np.mean(f1s_i), np.mean(accs_i), np.mean(recalls_i), np.mean(precisions_i))

#%%

# OVERSAMPLING INDUSTRIES


#%%

def svc_sm_notun_ind(df, df_t):
    
    X_train = df.iloc[:,:26]
    y_train = df.iloc[:,26:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    svc = svm.SVC(kernel='rbf')
    svc.fit(Xsm_train, ysm_train)

    predictions = svc.predict(df_t.iloc[:,:26])

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_smi = [[] for i in range(15)]
y_tests_smi = [[] for i in range(15)]
f1s_smi = [None for i in range(15)]
accs_smi = [None for i in range(15)]
recalls_smi = [None for i in range(15)]
precisions_smi = [None for i in range(15)]

for i in list(range(15)):
    predictions_smi[i], y_tests_smi[i], f1s_smi[i], accs_smi[i], recalls_smi[i], precisions_smi[i] = svc_sm_notun_ind(df_ind[i], df_tind[i])

#%%

for i in list(range(15)):
    y_tests_smi[i] = y_tests_smi[i].reshape(y_tests_smi[i].shape[0],1)
    predictions_smi[i] = predictions_smi[i].reshape(predictions_smi[i].shape[0],1)
    
#%%
predict_cm_smi = confusion_matrix(np.vstack(y_tests_smi), np.vstack(predictions_smi))
actual_cm = confusion_matrix(np.vstack(y_tests_smi), np.vstack(y_tests_smi))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_smi, labels, title="SVM w/ Oversampling w/o Tuning (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SMI (f1, acc, recall, precision):', np.mean(f1s_smi), np.mean(accs_smi), np.mean(recalls_smi), np.mean(precisions_smi))

#%%

# INDUSTRIES TUNING


#%%

def svc_nosm_tun_ind(df, df_t):
    
    svc = svm.SVC(C=1, gamma=0.01, kernel='rbf', class_weight={0:4, 1:96})
    svc.fit(df.iloc[:,:26], df.iloc[:,26:])

    predictions = svc.predict(df_t.iloc[:,:26])

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_tni = [[] for i in range(15)]
y_tests_tni = [[] for i in range(15)]
f1s_tni = [None for i in range(15)]
accs_tni = [None for i in range(15)]
recalls_tni = [None for i in range(15)]
precisions_tni = [None for i in range(15)]
for i in list(range(15)):
    predictions_tni[i], y_tests_tni[i], f1s_tni[i], accs_tni[i], recalls_tni[i], precisions_tni[i] = svc_nosm_tun_ind(df_ind[i], df_tind[i])

#%%

for i in list(range(15)):
    y_tests_tni[i] = y_tests_tni[i].reshape(y_tests_tni[i].shape[0],1)
    predictions_tni[i] = predictions_tni[i].reshape(predictions_tni[i].shape[0],1)

#%%
predict_cm_tni = confusion_matrix(np.vstack(y_tests_tni), np.vstack(predictions_tni))
actual_cm = confusion_matrix(np.vstack(y_tests_tni), np.vstack(y_tests_tni))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_tni, labels, title="SVM w/o Oversampling w/ Tuning (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('TUNI (f1, acc, recall, precision):', np.mean(f1s_tni), np.mean(accs_tni), np.mean(recalls_tni), np.mean(precisions_tni))

#%%

# OVERSAMPLING + TUNING INDUSTRIES


#%%

def svc_sm_tun_ind(df, df_t):
    
    X_train = df.iloc[:,:26]
    y_train = df.iloc[:,26:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    svc = svm.SVC(C=1, gamma=0.01, kernel='rbf', class_weight={0:4, 1:96})
    svc.fit(Xsm_train, ysm_train)

    predictions = svc.predict(df_t.iloc[:,:26])

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_smtni = [[] for i in range(15)]
y_tests_smtni = [[] for i in range(15)]
f1s_smtni = [None for i in range(15)]
accs_smtni = [None for i in range(15)]
recalls_smtni = [None for i in range(15)]
precisions_smtni = [None for i in range(15)]

for i in list(range(15)):
    predictions_smtni[i], y_tests_smtni[i], f1s_smtni[i], accs_smtni[i], recalls_smtni[i], precisions_smtni[i] = svc_sm_notun_ind(df_ind[i], df_tind[i])

#%%

for i in list(range(15)):
    y_tests_smtni[i] = y_tests_smtni[i].reshape(y_tests_smtni[i].shape[0],1)
    predictions_smtni[i] = predictions_smtni[i].reshape(predictions_smtni[i].shape[0],1)
    
#%%
predict_cm_smtni = confusion_matrix(np.vstack(y_tests_smtni), np.vstack(predictions_smtni))
actual_cm = confusion_matrix(np.vstack(y_tests_smtni), np.vstack(y_tests_smtni))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_smtni, labels, title="SVM w/ Oversampling w/ Tuning (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SMTUNI (f1, acc, recall, precision):', np.mean(f1s_smtni), np.mean(accs_smtni), np.mean(recalls_smtni), np.mean(precisions_smtni))