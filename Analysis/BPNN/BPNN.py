#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:14:13 2020

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

def nn_nosm(df, df_t):
    
    n_inputs = df.iloc[:,:16].shape[1]

    nn = Sequential([
        Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
        ])

    nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[Accuracy(), Recall()])

    nn.fit(df.iloc[:,:16], df.iloc[:,16:], validation_split=0.2,batch_size=100, epochs=50, shuffle=True, verbose=2)

    predictions = nn.predict_classes(df_t.iloc[:,:16], verbose=0)

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    #Metrics
    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    #roc_auc = roc_auc_score(y_test, predictions)

    return predictions, y_test, f1, acc, recall, precision

#%%
predictions = [[] for i in range(15)]
y_tests = [[] for i in range(15)]
f1s = [None for i in range(15)]
accs = [None for i in range(15)]
recalls = [None for i in range(15)]
precisions = [None for i in range(15)]
for i in list(range(15)):
    predictions[i], y_tests[i], f1s[i], accs[i], recalls[i], precisions[i] = nn_nosm(df_list[i], df_test[i])

#%%

for i in list(range(15)):
    y_tests[i] = y_tests[i].reshape(y_tests[i].shape[0],1)
    
#%%
predict_cm = confusion_matrix(np.vstack(y_tests), np.vstack(predictions))
actual_cm = confusion_matrix(np.vstack(y_tests), np.vstack(y_tests))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm, labels, title="BPNN w/o Oversampling \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%% 

# OVERSAMPLING


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

    predictions = nn.predict_classes(df_t.iloc[:,:16], verbose=0)

    #probas = nn.predict_proba(df_t.iloc[:,:16])

    y_test = df_t.iloc[:,16:]['y_t+2'].values

    #Metrics
    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    #roc_auc = roc_auc_score(y_test, predictions)

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
    predictions_sm[i], y_tests_sm[i], f1s_sm[i], accs_sm[i], recalls_sm[i], precisions_sm[i] = nn_sm(df_list[i], df_test[i])

#%%
"""
import copy
threshold = 0.4
predprob_sm = copy.deepcopy(predictions_sm)

for i in list(range(15)):
    for j in list(range(len(probas_sm[i]))):
        prob = probas_sm[i][j][0]
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
    f1s_prob_sm[i] = f1_score(y_tests_sm[i], predprob_sm[i])
    accs_prob_sm[i] = accuracy_score(y_tests_sm[i], predprob_sm[i])
    recalls_prob_sm[i] = recall_score(y_tests_sm[i], predprob_sm[i])
    precisions_prob_sm[i] = precision_score(y_tests_sm[i], predprob_sm[i])
"""            
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


#%%
"""
predict_cm_smprob = confusion_matrix(np.vstack(y_tests_sm), np.vstack(predprob_sm))
actual_cm = confusion_matrix(np.vstack(y_tests_sm), np.vstack(y_tests_sm))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_smprob, labels, title="BPNN w/ Oversampling (w/ threshold) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

print('SM (w/ threshold) (f1, acc, recall, precision):', np.mean(f1s_prob_sm), np.mean(accs_prob_sm), np.mean(recalls_prob_sm), np.mean(precisions_prob_sm))
"""
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

def nn_nosmind(df, df_t):
    
    n_inputs = df.iloc[:,:26].shape[1]

    nn = Sequential([
        Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
        ])

    nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[Accuracy(), Recall()])

    nn.fit(df.iloc[:,:26], df.iloc[:,26:], validation_split=0.2,batch_size=100, epochs=50, shuffle=True, verbose=2)

    predictions = nn.predict_classes(df_t.iloc[:,:26], verbose=0)

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    #Metrics
    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    #roc_auc = roc_auc_score(y_test, predictions)

    return predictions, y_test, f1, acc, recall, precision

#%%
predictions_i = [[] for i in range(15)]
y_tests_i = [[] for i in range(15)]
f1s_i = [None for i in range(15)]
accs_i = [None for i in range(15)]
recalls_i = [None for i in range(15)]
precisions_i = [None for i in range(15)]
for i in list(range(15)):
    predictions_i[i], y_tests_i[i], f1s_i[i], accs_i[i], recalls_i[i], precisions_i[i] = nn_nosm(df_list[i], df_test[i])

#%%

for i in list(range(15)):
    y_tests_i[i] = y_tests_i[i].reshape(y_tests_i[i].shape[0],1)
    
#%%
predict_cm_i = confusion_matrix(np.vstack(y_tests_i), np.vstack(predictions_i))
actual_cm = confusion_matrix(np.vstack(y_tests_i), np.vstack(y_tests_i))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_i, labels, title="BPNN w/o Oversampling (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)


#%%

# OVERSAMPLING INDUSTRIES
#from keras.wrappers.scikit_learn import KerasClassifier
#import eli5
#from eli5.sklearn import PermutationImportance

#%%

def nn_smind(df, df_t):
    
    X_train = df.iloc[:,:26]
    y_train = df.iloc[:,26:]

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)


    # This will be the data were we are going to 
    Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)
    
    n_inputs = df.iloc[:,:26].shape[1]

    nn = Sequential([
        Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
        ])

    nn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[Accuracy(), Recall()])

    nn.fit(Xsm_train, ysm_train, validation_split=0.2,batch_size=100, epochs=50, shuffle=True, verbose=2)

    predictions = nn.predict_classes(df_t.iloc[:,:26], verbose=0)

    #weights = nn.layers[0].get_weights()[0]

    y_test = df_t.iloc[:,26:]['y_t+2'].values

    #Metrics
    f1 = f1_score(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    #roc_auc = roc_auc_score(y_test, predictions)

    return predictions, y_test, f1, acc, recall, precision#, weights

#%%
predictions_smi = [[] for i in range(15)]
y_tests_smi = [[] for i in range(15)]
f1s_smi = [None for i in range(15)]
accs_smi = [None for i in range(15)]
recalls_smi = [None for i in range(15)]
precisions_smi = [None for i in range(15)]
#weights_smi = [None for i in range(15)]

for i in list(range(15)):
    predictions_smi[i], y_tests_smi[i], f1s_smi[i], accs_smi[i], recalls_smi[i], precisions_smi[i] = nn_smind(df_ind[i], df_tind[i])

#%%

for i in list(range(15)):
    y_tests_smi[i] = y_tests_smi[i].reshape(y_tests_smi[i].shape[0],1)
    
#%%
predict_cm_smi = confusion_matrix(np.vstack(y_tests_smi), np.vstack(predictions_smi))
actual_cm = confusion_matrix(np.vstack(y_tests_smi), np.vstack(y_tests_smi))
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_smi, labels, title="BPNN w/ Oversampling (Industries) \n Confusion Matrix", cmap=plt.cm.Blues)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

#%%

print('Basic (f1, acc, recall, precision):', np.mean(f1s), np.mean(accs), np.mean(recalls), np.mean(precisions))
print('SM (f1, acc, recall, precision):', np.mean(f1s_sm), np.mean(accs_sm), np.mean(recalls_sm), np.mean(precisions_sm))
print('I (f1, acc, recall, precision):', np.mean(f1s_i), np.mean(accs_i), np.mean(recalls_i), np.mean(precisions_i))
print('SMI (f1, acc, recall, precision):', np.mean(f1s_smi), np.mean(accs_smi), np.mean(recalls_smi), np.mean(precisions_smi))

#%%

rec_df = pd.DataFrame(list(zip(recalls, recalls_sm, recalls_i, recalls_smi)), columns = ['Simple', 'SM','IND','SMI'],
                      index=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])

#rec_df.to_csv('rec_df_nn.csv')

#%%
f1_df = pd.DataFrame(list(zip(f1s, f1s_sm, f1s_i, f1s_smi)), columns = ['Simple', 'SM', 'IND','SMI'],
                      index=['2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'])

#f1_df.to_csv('f1_df_nn.csv')

#%%
predict_cm_smi08 = confusion_matrix(y_tests_smi[3], predictions_smi[3])
actual_cm = confusion_matrix(y_tests_smi[3], y_tests_smi[3])
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_smi08, labels, title="BPNN SMI (2008) \n Confusion Matrix", cmap=plt.cm.Greys)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greys)

#%%
predict_cm_sm08 = confusion_matrix(y_tests_sm[3], predictions_sm[3])
actual_cm = confusion_matrix(y_tests_sm[3], y_tests_sm[3])
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm_sm08, labels, title="BPNN SM (2008) \n Confusion Matrix", cmap=plt.cm.Greys)

#fig.add_subplot(222)
#plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
