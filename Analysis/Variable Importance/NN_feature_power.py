#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 10:33:43 2020

@author: evaferreira
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# Import Neural Network Library
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.metrics import Accuracy, Recall 

# Other Libraries
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
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
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

def nn_sm(df, df_t):
    
    X_train = df.iloc[:,:16]
    y_train = df.iloc[:,16:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    n_inputs = df.iloc[:,:16].shape[1]

    nn = Sequential([
        Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
        ])

    nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[Accuracy(), Recall()])

    nn.fit(Xsm_train, ysm_train, validation_split=0.2,batch_size=100, epochs=50, shuffle=True, verbose=2)
   
    df_t.iloc[:,6] = np.random.permutation(df_t.iloc[:,6].values)
    
    predictions = nn.predict_classes(df_t.iloc[:,:16], verbose=0)

    probas = nn.predict_proba(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    #Metrics
    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    #roc_auc = roc_auc_score(y_test, predictions)

    return predictions, y_test, f1, acc, recall, precision, probas


#%%
predictions_sm = [[] for i in range(15)]
y_tests_sm = [[] for i in range(15)]
f1s_sm = [None for i in range(15)]
accs_sm = [None for i in range(15)]
recalls_sm = [None for i in range(15)]
precisions_sm = [None for i in range(15)]
probas_sm = [None for i in range(15)]

for i in list(range(15)):
    predictions_sm[i], y_tests_sm[i], f1s_sm[i], accs_sm[i], recalls_sm[i], precisions_sm[i], probas_sm[i] = nn_sm(df_list[i], df_test[i])
 
#%%

for i in list(range(15)):
    y_tests_sm[i] = y_tests_sm[i].reshape(y_tests_sm[i].shape[0],1)
    
#%%
predict_cm_sm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(predictions_sm))
actual_cm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(y_tests_sm))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_sm, labels, title="BPNN w/ Oversampling \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SM (f1, acc, recall, precision):', np.mean(f1s_sm), np.mean(accs_sm), np.mean(recalls_sm), np.mean(precisions_sm))
