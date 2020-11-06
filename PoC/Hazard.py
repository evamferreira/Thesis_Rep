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

dfo = pd.read_csv('/Users/evaferreira/Downloads/Thesis/Thesis_Rep/Data/dfo.csv')
dfo.drop(columns=['Unnamed: 0'], inplace=True)

#%%

# TRAIN DFs


#%%

def clean_df(dfo0x, year_list):
    # Summing the variables
    dfo0x['sumvar'] = dfo0x.iloc[:, -8:-1].sum(axis=1)
    
    # Cumulative Sum of variables throughout the years per PERMCO
    dfo0x['sumvar'] = dfo0x.groupby('PERMCO').transform(sum).sumvar
    
    # Dropping the companies with all 0s in variable,s (either not created yet or bankrupt for all of the timeframe)
    dropsum0 = dfo0x[dfo0x.sumvar==0]['PERMCO'].unique()
    
    # Drop companies from dfo with sum of var for all 5 years as 0
    for i in dropsum0:
        dfo0x.drop(dfo0x.index[dfo0x['PERMCO']==i], inplace=True)
    
    dfo0x.drop(columns='sumvar', inplace=True)
    
    dfo0x['dlrsn_lag'] = dfo0x.groupby(['PERMCO'])['dlrsn'].shift(-1)
    
    dfo0x.drop(dfo0x[dfo0x.fyear.isin([str(year_list[4]+1)])].index, inplace=True)
    
    return dfo0x

#%%

def create_finaldf(dfo0x, year_list):
    NIMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'NIMTA']]
    TLMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'TLMTA']]
    CASHMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'CASHMTA']]
    MB = dfo0x.loc[:, ['PERMCO', 'fyear', 'MB']]
    SIGMA = dfo0x.loc[:, ['PERMCO', 'fyear', 'SIGMA']]
    RSIZE = dfo0x.loc[:, ['PERMCO', 'fyear', 'RSIZE']]
    EXRET = dfo0x.loc[:, ['PERMCO', 'fyear', 'EXRET']]
    PRICE = dfo0x.loc[:, ['PERMCO', 'fyear', 'PRICE']]
    
    varstr = ['NIMTA', 'TLMTA', 'CASHMTA', 'MB', 'SIGMA', 'RSIZE', 'EXRET', 'PRICE']
    var = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]
    varcols = {c: [] for c in varstr}
    
    for i in list(range(8)):
        var[i] = var[i].set_index(['PERMCO','fyear'])[varstr[i]].unstack()
        for y in year_list:
            varcols[varstr[i]].append('{}_{}'.format(varstr[i], y))
        var[i].columns = varcols[varstr[i]]
    
    var_0x1 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]
    var_0x2 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]
    var_0x3 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]
    var_0x4 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]
    var_yrs = [var_0x1,var_0x2,var_0x3,var_0x4]
    
    for j in list(range(4)):
        for i in list(range(8)):
            var_yrs[j][i] = var[i].iloc[:,[j,j+1]]
            
    df_yrs = [var_yrs[0][0], var_yrs[1][0], var_yrs[2][0], var_yrs[3][0]]
    
    for j in list(range(4)):
        for df_ in var_yrs[j][1:]:
            df_yrs[j] = df_yrs[j].merge(df_, on='PERMCO')
    
    col_names = ['NIMTA_t','NIMTA_t+1','TLMTA_t','TLMTA_t+1','CASHMTA_t','CASHMTA_t+1','MB_t','MB_t+1','SIGMA_t',
             'SIGMA_t+1','RSIZE_t','RSIZE_t+1','EXRET_t','EXRET_t+1','PRICE_t','PRICE_t+1']
    
    for j in list(range(4)):
        df_yrs[j].columns = col_names
        df_yrs[j]['PERMCO_Y'] = df_yrs[j].index.astype(str) + '_' + str(year_list[j]+2)
        df_yrs[j].set_index('PERMCO_Y', inplace=True)
        
    df_yrs[0] = df_yrs[0].append(df_yrs[1])
    df_yrs[0] = df_yrs[0].append(df_yrs[2])
    df_yrs[0] = df_yrs[0].sort_index()
    
    return df_yrs[0]


#%%

def append_dlrsn(dfo0x, df_listobj, year_list):
    dlrsn0x = dfo0x[['PERMCO', 'fyear', 'dlrsn_lag']]
    dlrsn0x = dlrsn0x[dlrsn0x.fyear != year_list[0]]
    dlrsn0x = dlrsn0x[dlrsn0x.fyear != year_list[4]]
    dlrsn0x['PERMCO_Y'] = dlrsn0x['PERMCO'].astype(str) + '_' + (dlrsn0x['fyear']+1).astype(str)
    dlrsn0x.set_index('PERMCO_Y', inplace=True)
    dlrsn0x.drop(columns=['PERMCO', 'fyear'], inplace=True)
    dlrsn0x = dlrsn0x.sort_index()
    df_listobj['y_t+2'] = dlrsn0x
    
    return df_listobj


#%%

year_lists = [[] for i in range(15)]
for i in list(range(15)):
    year_lists[i] = list(range(2000+i, 2005+i, +1))
    
#%%

ogdf_list = [pd.DataFrame() for i in range(15)]
for i in list(range(15)):
    ogdf_list[i] = dfo[(dfo['fyear']>=2000+i) & (dfo['fyear']<=2005+i)]

#%%

for i in list(range(15)):
    ogdf_list[i] = clean_df(ogdf_list[i], year_lists[i])
    
#%%

df_list = [pd.DataFrame() for i in range(15)]
for i in list(range(15)):
    df_list[i] = create_finaldf(ogdf_list[i], year_lists[i])
    
#%%

for i in list(range(15)):
    df_list[i] = append_dlrsn(ogdf_list[i], df_list[i], year_lists[i])
    df_list[i] = df_list[i].sort_index()
    
#%%

# TEST DFs


#%%

def clean_dftest(dfo0x, year_list):
    # Summing the variables
    dfo0x['sumvar'] = dfo0x.iloc[:, -8:-1].sum(axis=1)
    
    # Cumulative Sum of variables throughout the years per PERMCO
    dfo0x['sumvar'] = dfo0x.groupby('PERMCO').transform(sum).sumvar
    
    # Dropping the companies with all 0s in variable,s (either not created yet or bankrupt for all of the timeframe)
    dropsum0 = dfo0x[dfo0x.sumvar==0]['PERMCO'].unique()
    
    # Drop companies from dfo with sum of var for all 3 years as 0
    for i in dropsum0:
        dfo0x.drop(dfo0x.index[dfo0x['PERMCO']==i], inplace=True)
    
    dfo0x.drop(columns='sumvar', inplace=True)
    
    dfo0x['dlrsn_lag'] = dfo0x.groupby(['PERMCO'])['dlrsn'].shift(-1)

    dfo0x.drop(dfo0x[dfo0x.fyear.isin([str(year_list[0]),str(year_list[1]),str(year_list[2])])].index, inplace=True)
    
    return dfo0x

#%%

def create_finaldftest(dfo0x, year_list):
    NIMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'NIMTA']]
    TLMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'TLMTA']]
    CASHMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'CASHMTA']]
    MB = dfo0x.loc[:, ['PERMCO', 'fyear', 'MB']]
    SIGMA = dfo0x.loc[:, ['PERMCO', 'fyear', 'SIGMA']]
    RSIZE = dfo0x.loc[:, ['PERMCO', 'fyear', 'RSIZE']]
    EXRET = dfo0x.loc[:, ['PERMCO', 'fyear', 'EXRET']]
    PRICE = dfo0x.loc[:, ['PERMCO', 'fyear', 'PRICE']]
    
    varstr = ['NIMTA', 'TLMTA', 'CASHMTA', 'MB', 'SIGMA', 'RSIZE', 'EXRET', 'PRICE']
    var = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]
    varcols = {c: [] for c in varstr}
    
    for i in list(range(8)):
        var[i] = var[i].set_index(['PERMCO','fyear'])[varstr[i]].unstack()
        for y in year_list:
            varcols[varstr[i]].append('{}_{}'.format(varstr[i], y))
        var[i].columns = varcols[varstr[i]]
    
    var_0x1 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE]

    for i in list(range(8)):
        var_0x1[i] = var[i].iloc[:,[0,1]]
     
    df_vars = var_0x1[0]
    
    for df_ in var_0x1[1:]:
        df_vars = df_vars.merge(df_, on='PERMCO')
    
    col_names = ['NIMTA_t','NIMTA_t+1','TLMTA_t','TLMTA_t+1','CASHMTA_t','CASHMTA_t+1','MB_t','MB_t+1','SIGMA_t',
             'SIGMA_t+1','RSIZE_t','RSIZE_t+1','EXRET_t','EXRET_t+1','PRICE_t','PRICE_t+1']
    
    df_vars.columns = col_names
    df_vars['PERMCO_Y'] = df_vars.index.astype(str) + '_' + str(year_list[0]+2)
    df_vars.set_index('PERMCO_Y', inplace=True)
        
    df_vars = df_vars.sort_index()
    
    return df_vars

#%%

def append_dlrsntest(dfo0x, df_listobj, year_list):
    dlrsn0x = dfo0x[['PERMCO', 'fyear', 'dlrsn_lag']]
    dlrsn0x = dlrsn0x[dlrsn0x.fyear == year_list[4]]
    dlrsn0x['PERMCO_Y'] = dlrsn0x['PERMCO'].astype(str) + '_' + (dlrsn0x['fyear']+1).astype(str)
    dlrsn0x.set_index('PERMCO_Y', inplace=True)
    dlrsn0x.drop(columns=['PERMCO', 'fyear'], inplace=True)
    dlrsn0x = dlrsn0x.sort_index()
    df_listobj['y_t+2'] = dlrsn0x
    
    return df_listobj

#%%

ogdf_test = [pd.DataFrame() for i in range(15)]
for i in list(range(15)):
    ogdf_test[i] = dfo[(dfo['fyear']>=2003+i) & (dfo['fyear']<=2005+i)]
    
#%%

for i in list(range(15)):
    ogdf_test[i] = clean_df(ogdf_test[i], year_lists[i])
    
#%%

df_test = [pd.DataFrame() for i in range(15)]
for i in list(range(15)):
    df_test[i] = create_finaldftest(ogdf_test[i], year_lists[i][3:5])
    
#%%

for i in list(range(15)):
    df_test[i] = append_dlrsntest(ogdf_test[i], df_test[i], year_lists[i])
    df_test[i] = df_test[i].sort_index()

#%%

# LOGISTIC REGRESSION

    
#%%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

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
print(f1, acc, recall, precision)

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

predict_cm = confusion_matrix(y_test, predictions)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Bankruptcy', 'Bankruptcy']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predict_cm, labels, title="Predictions \n Confusion Matrix", cmap=plt.cm.Oranges)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
