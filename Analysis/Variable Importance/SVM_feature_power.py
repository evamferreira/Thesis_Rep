#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:05:29 2020

@author: evaferreira
"""
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

# Rescale the data
from sklearn.preprocessing import RobustScaler

# RobustScaler is less prone to outliers.

rob_scaler = RobustScaler()

for df_ in list(range(len(df_list))):
    for i in df_list[df_].iloc[:,:16].columns.to_list():    
        df_list[df_].iloc[:,:16][i] = rob_scaler.fit_transform(df_list[df_].iloc[:,:16][i].values.reshape(-1,1))  
        df_test[df_].iloc[:,:16][i] = rob_scaler.fit_transform(df_test[df_].iloc[:,:16][i].values.reshape(-1,1))

#%%

# TUNING 
from sklearn.model_selection import GridSearchCV

#%%

def tuning(df,i=False):

    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 
                  'class_weight': [{0:1, 1:1,}, {0:4,1:96}, {0:1, 1:100}]}

    grid = GridSearchCV(svm.SVC(),param_grid, scoring="f1", verbose=2)
    if i==True:
        grid.fit(df.iloc[:,:26], df.iloc[:,26:])
    else:
        grid.fit(df.iloc[:,:16], df.iloc[:,16:])

    return grid.best_params_
#%%
best_ps = [None for i in range(15)]

for i in list(range(15)):
    best_ps[i] = tuning(df_list[i])
    
#%%

def svc_nosm_tun(df, df_t, best_p):
    
    svc = svm.SVC(**best_p)
    svc.fit(df.iloc[:,:16], df.iloc[:,16:])
    
    df_t.iloc[:,15] = np.random.permutation(df_t.iloc[:,15].values)
    
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
    predictions_tn[i], y_tests_tn[i], f1s_tn[i], accs_tn[i], recalls_tn[i], precisions_tn[i] = svc_nosm_tun(df_list[i], df_test[i], best_ps[i])

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

