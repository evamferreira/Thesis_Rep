#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:59:35 2020

@author: evaferreira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
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
                          normalize=True,
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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

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

def logreg_nosm_nowc(df, df_t):
    
    logreg = LogisticRegression()

    logreg.fit(df.iloc[:,:16], df.iloc[:,16:])
    
    predictions = logreg.predict(df_t.iloc[:,:16])

    #weights = logreg.coef_[0]

    #probas = logreg.predict_proba(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    return predictions, y_test, f1, acc, recall, precision#, probas, weights

#%%
predictions = [[] for i in range(15)]
y_tests = [None for i in range(15)]
f1s = [None for i in range(15)]
accs = [None for i in range(15)]
recalls = [None for i in range(15)]
precisions = [None for i in range(15)]
probas = [None for i in range(15)]
#weights = [None for i in range(15)]

for i in list(range(15)):
    predictions[i], y_tests[i], f1s[i], accs[i], recalls[i], precisions[i] = logreg_nosm_nowc(df_list[i], df_test[i])

#%%
"""import copy
threshold = 0.4
predprob = copy.deepcopy(predictions)

for i in list(range(15)):
    for j in list(range(len(probas[i]))):
        prob = probas[i][j][1]
        if prob >= threshold:
            predprob[i][j] = 1.0
        else:
            predprob[i][j] = 0.0

#%%
f1s_prob = [None for i in range(15)]
accs_prob = [None for i in range(15)]
recalls_prob = [None for i in range(15)]
precisions_prob = [None for i in range(15)]

for i in list(range(15)):
    f1s_prob[i] = f1_score(y_tests[i], predprob[i])
    accs_prob[i] = accuracy_score(y_tests[i], predprob[i])
    recalls_prob[i] = recall_score(y_tests[i], predprob[i])
    precisions_prob[i] = precision_score(y_tests[i], predprob[i])
 """           
#%%

for i in list(range(15)):
    y_tests[i] = y_tests[i].reshape(y_tests[i].shape[0],1)
    predictions[i] = predictions[i].reshape(predictions[i].shape[0],1)
    #predprob[i] = predprob[i].reshape(predprob[i].shape[0],1)
    

#%%
predict_cm = confusion_matrix(np.vstack(y_tests), np.vstack(predictions))
actual_cm = confusion_matrix(np.vstack(y_tests), np.vstack(y_tests))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm, labels, title="DTH w/o Oversampling w/o Weighed-Class \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('Simple Logit (f1, acc, recall, precision):', np.mean(f1s), np.mean(accs), np.mean(recalls), np.mean(precisions))

#%%
"""predict_cm_prob = confusion_matrix(np.vstack(y_tests), np.vstack(predprob))
actual_cm = confusion_matrix(np.vstack(y_tests), np.vstack(y_tests))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_prob, labels, title="DTH w/o Oversampling w/o Weighed-Class (With dif threshold) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('Basic w/ threshold (f1, acc, recall, precision):', np.mean(f1s_prob), np.mean(accs_prob), np.mean(recalls_prob), np.mean(precisions_prob))

#%%

argmax = [None for i in range(15)]

for i in list(range(15)):
    argmax[i] = np.argmax(np.absolute(weights[i]))
"""
#%%

#WEIGHTED CLASS LOGREG
from sklearn.model_selection import GridSearchCV

#%%

def tuning(df,i=False):

    gs_w = [{0:1000,1:100},{0:1000,1:10}, {0:1000,1:1.0}, 
     {0:500,1:1.0}, {0:400,1:1.0}, {0:300,1:1.0}, {0:200,1:1.0}, 
     {0:150,1:1.0}, {0:100,1:1.0}, {0:99,1:1.0}, {0:10,1:1.0}, 
     {0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100}, 
     {0:0.001,1:1.0}, {0:0.005,1:1.0}, {0:1.0,1:1.0}, 
     {0:1.0,1:0.1}, {0:10,1:0.1}, {0:100,1:0.1}, 
     {0:10,1:0.01}, {0:1.0,1:0.01}, {0:1.0,1:0.001}, {0:1.0,1:0.005}, 
     {0:1.0,1:10}, {0:1.0,1:99}, {0:1.0,1:100}, {0:1.0,1:150}, 
     {0:1.0,1:200}, {0:1.0,1:300},{0:1.0,1:400},{0:1.0,1:500}, 
     {0:1.0,1:1000}, {0:10,1:1000},{0:100,1:1000}, {0:4,1:96} ]

    param_grid = {"class_weight": gs_w }

    grid = GridSearchCV(LogisticRegression(),param_grid, scoring="f1", verbose=2)
    
    if i==True:
        grid.fit(df.iloc[:,:26], df.iloc[:,26:])
    else:
        grid.fit(df.iloc[:,:16], df.iloc[:,16:])

    return grid.best_params_

#%%

def logreg_nosm_wc(df, df_t):
    
    best_p = tuning(df)
    
    logreg = LogisticRegression(**best_p)

    logreg.fit(df.iloc[:,:16], df.iloc[:,16:])

    predictions = logreg.predict(df_t.iloc[:,:16])

    #weights = logreg.coef_[0]

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_wc = [[] for i in range(15)]
y_tests_wc = [None for i in range(15)]
f1s_wc = [None for i in range(15)]
accs_wc = [None for i in range(15)]
recalls_wc = [None for i in range(15)]
precisions_wc = [None for i in range(15)]

for i in list(range(15)):
    predictions_wc[i], y_tests_wc[i], f1s_wc[i], accs_wc[i], recalls_wc[i], precisions_wc[i] = logreg_nosm_wc(df_list[i], df_test[i])
    print('///////////////', i)
    
#%%

for i in list(range(15)):
    y_tests_wc[i] = y_tests_wc[i].reshape(y_tests_wc[i].shape[0],1)
    predictions_wc[i] = predictions_wc[i].reshape(predictions_wc[i].shape[0],1)

#%%
predict_cm_wc = confusion_matrix(np.vstack(y_tests_wc), np.vstack(predictions_wc))
actual_cm = confusion_matrix(np.vstack(y_tests_wc), np.vstack(y_tests_wc))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_wc, labels, title="DTH w/o Oversampling w/ Weighted-Class \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('WC (f1, acc, recall, precision):', np.mean(f1s_wc), np.mean(accs_wc), np.mean(recalls_wc), np.mean(precisions_wc))

#%%
 
# OVERSAMPLING

#%%

def logreg_sm_nowc(df, df_t):
    
    X_train = df.iloc[:,:16]
    y_train = df.iloc[:,16:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    w = {0: 0.01, 1: 1.0}
    logreg = LogisticRegression(class_weight=w)

    logreg.fit(Xsm_train, ysm_train)

    predictions = logreg.predict(df_t.iloc[:,:16])

    #probas = logreg.predict_proba(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    return predictions, y_test, f1, acc, recall, precision#, probas

#%%
predictions_sm = [[] for i in range(15)]
y_tests_sm = [[] for i in range(15)]
f1s_sm = [None for i in range(15)]
accs_sm = [None for i in range(15)]
recalls_sm = [None for i in range(15)]
precisions_sm = [None for i in range(15)]
#probas_sm = [None for i in range(15)]

for i in list(range(15)):
    predictions_sm[i], y_tests_sm[i], f1s_sm[i], accs_sm[i], recalls_sm[i], precisions_sm[i] = logreg_sm_nowc(df_list[i], df_test[i])

#%%
"""
import copy
threshold = 0.9
predprob_sm = copy.deepcopy(predictions_sm)

for i in list(range(15)):
    for j in list(range(len(probas_sm[i]))):
        prob = probas_sm[i][j][1]
        if prob >= threshold:
            predprob_sm[i][j] = 1.0
        else:
            predprob_sm[i][j] = 0.0
 
#%%

f1s_prob_sm = [None for i in range(15)]
accs_prob_sm = [None for i in range(15)]
recalls_prob_sm = [None for i in range(15)]
precisions_prob_sm = [None for i in range(15)]

for i in list(range(15)):
    f1s_prob_sm[i] = f1_score(y_tests[i], predprob_sm[i])
    accs_prob_sm[i] = accuracy_score(y_tests[i], predprob_sm[i])
    recalls_prob_sm[i] = recall_score(y_tests[i], predprob_sm[i])
    precisions_prob_sm[i] = precision_score(y_tests[i], predprob_sm[i])
"""            
#%%

for i in list(range(15)):
    y_tests_sm[i] = y_tests_sm[i].reshape(y_tests_sm[i].shape[0],1)
    predictions_sm[i] = predictions_sm[i].reshape(predictions_sm[i].shape[0],1)
    #predprob_sm[i] = predprob_sm[i].reshape(predprob_sm[i].shape[0],1)
    
#%%
predict_cm_sm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(predictions_sm))
actual_cm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(y_tests_sm))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_sm, labels, title="DTH w/ Oversampling w/o Weighted-Class \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SM (f1, acc, recall, precision):', np.mean(f1s_sm), np.mean(accs_sm), np.mean(recalls_sm), np.mean(precisions_sm))

#%%
"""
predict_cm_smprob = confusion_matrix(np.vstack(y_tests_sm), np.vstack(predprob_sm))
actual_cm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(y_tests_sm))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_smprob, labels, title="DTH w/ Oversampling w/o Weighted-Class (w/ threshold) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SM (w/ threshold) (f1, acc, recall, precision):', np.mean(f1s_prob_sm), np.mean(accs_prob_sm), np.mean(recalls_prob_sm), np.mean(precisions_prob_sm))
"""
#%%

# INDUSTRY EFFECTS

#%%

def logreg_nosm_nowc_ind(df, df_t):
    
    #with weighted class recall is higher but f1 is worse
    w = {0: 0.01, 1: 1.0}
    logreg = LogisticRegression(class_weight=w)

    logreg.fit(df.iloc[:,:26], df.iloc[:,26:])

    predictions = logreg.predict(df_t.iloc[:,:26])

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    return predictions, y_test, f1, acc, recall, precision
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
predictions_i = [[] for i in range(15)]
y_tests_i = [None for i in range(15)]
f1s_i = [None for i in range(15)]
accs_i = [None for i in range(15)]
recalls_i = [None for i in range(15)]
precisions_i = [None for i in range(15)]

for i in list(range(15)):
    predictions_i[i], y_tests_i[i], f1s_i[i], accs_i[i], recalls_i[i], precisions_i[i] = logreg_nosm_nowc_ind(df_ind[i], df_tind[i])

#%%

for i in list(range(15)):
    y_tests_i[i] = y_tests_i[i].reshape(y_tests_i[i].shape[0],1)
    predictions_i[i] = predictions_i[i].reshape(predictions_i[i].shape[0],1)

#%%
predict_cm_i = confusion_matrix(np.vstack(y_tests_i), np.vstack(predictions_i))
actual_cm = confusion_matrix(np.vstack(y_tests_i), np.vstack(y_tests_i))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_i, labels, title="DTH w/o Oversampling w/o Weighted-Class (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('I (f1, acc, recall, precision):', np.mean(f1s_i), np.mean(accs_i), np.mean(recalls_i), np.mean(precisions_i))

#%%

# INDUSTRIES OVERSAMPLING

#%%

def logreg_sm_nowc_ind(df, df_t):
    
    X_train = df.iloc[:,:26]
    y_train = df.iloc[:,26:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)

    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    w = {0: 0.01, 1: 1.0}
    logreg = LogisticRegression(class_weight=w)

    logreg.fit(Xsm_train, ysm_train)

    predictions = logreg.predict(df_t.iloc[:,:26])

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
    predictions_smi[i], y_tests_smi[i], f1s_smi[i], accs_smi[i], recalls_smi[i], precisions_smi[i] = logreg_sm_nowc_ind(df_ind[i], df_tind[i])

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
plot_confusion_matrix(predict_cm_smi, labels, title="DTH w/ Oversampling w/o Weighted Class (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SMI (f1, acc, recall, precision):', np.mean(f1s_smi), np.mean(accs_smi), np.mean(recalls_smi), np.mean(precisions_smi))
#%%

# INDUSTRY WEIGHTED CLASS

#%%

def logreg_nosm_wc_ind(df, df_t):
    
    best_p = tuning(df,True)
    
    logreg = LogisticRegression(**best_p)

    logreg.fit(df.iloc[:,:26], df.iloc[:,26:])

    predictions = logreg.predict(df_t.iloc[:,:26])

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    return predictions, y_test, f1, acc, recall, precision
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
predictions_wci = [[] for i in range(15)]
y_tests_wci = [None for i in range(15)]
f1s_wci = [None for i in range(15)]
accs_wci = [None for i in range(15)]
recalls_wci = [None for i in range(15)]
precisions_wci = [None for i in range(15)]

for i in list(range(15)):
    predictions_wci[i], y_tests_wci[i], f1s_wci[i], accs_wci[i], recalls_wci[i], precisions_wci[i] = logreg_nosm_wc_ind(df_ind[i], df_tind[i])
    print('/////////////////',i)
#%%

for i in list(range(15)):
    y_tests_wci[i] = y_tests_wci[i].reshape(y_tests_wci[i].shape[0],1)
    predictions_wci[i] = predictions_wci[i].reshape(predictions_wci[i].shape[0],1)

#%%
predict_cm_wci = confusion_matrix(np.vstack(y_tests_wci), np.vstack(predictions_wci))
actual_cm = confusion_matrix(np.vstack(y_tests_wci), np.vstack(y_tests_wci))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_wci, labels, title="DTH w/o Oversampling w/ Weighted-Class (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('WCI (f1, acc, recall, precision):', np.mean(f1s_wci), np.mean(accs_wci), np.mean(recalls_wci), np.mean(precisions_wci))

#%%

rec_df = pd.DataFrame(list(zip(recalls, recalls_wc, recalls_sm, recalls_i, recalls_wci, recalls_smi)), columns = ['Basic', 'WC', 'SM', 'Ind','WCI','SMI'],
                      index=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])

#rec_df.to_csv('rec_df_logit.csv')

#%%
f1_df = pd.DataFrame(list(zip(f1s, f1s_wc, f1s_sm, f1s_i, f1s_wci, f1s_smi)), columns = ['Basic', 'WC', 'SM', 'Ind','WCI','SMI'],
                      index=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])

#f1_df.to_csv('f1_df_logit.csv')

#%%
predict_cm_08 = confusion_matrix(y_tests[3], predictions[3])
actual_cm = confusion_matrix(y_tests[3], y_tests[3])
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_08, labels, title="Simple Logit (2008) \n Confusion Matrix", cmap=plt.cm.Greys)

#fig.add_subplot(222)
#plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greys)

#%%
predict_cm_wci08 = confusion_matrix(y_tests_wci[3], predictions_wci[3])
actual_cm = confusion_matrix(y_tests_wci[3], y_tests_wci[3])
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_wci08, labels, title="Logit TUNI (2008) \n Confusion Matrix", cmap=plt.cm.Greys)

#fig.add_subplot(222)
#plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greys)

#%%
predict_cm_sm08 = confusion_matrix(y_tests_sm[3], predictions_sm[3])
actual_cm = confusion_matrix(y_tests_sm[3], y_tests_sm[3])
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_sm08, labels, title="Logit SM (2008) \n Confusion Matrix", cmap=plt.cm.Greys)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greys)

