#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:35:15 2020

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

for col in pd.get_dummies(dfo['gind'], prefix='gind').columns:
    dfo[col] = pd.get_dummies(dfo['gind'], prefix='gind')[col]

dfo.drop(columns=['gind_10'], inplace=True)
    
#%%

# TRAIN DFs


#%%

def clean_df(dfo0x, year_list):
    # Summing the variables
    dfo0x['sumvar'] = dfo0x.iloc[:, 6:14].sum(axis=1)
    
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
    print(i)
    
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
    dfo0x['sumvar'] = dfo0x.iloc[:, 6:14].sum(axis=1)
    
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

#for i in list(range(15)):
#   df_list[i].to_csv('dftrain' + str(i) + '.csv')

#%%

#for i in list(range(15)): 
#   df_test[i].to_csv('dftest' + str(i) + '.csv')

#%%

# INDUSTRY DUMMIES

#%%

for col in pd.get_dummies(dfo['gind'], prefix='gind').columns:
    dfo[col] = pd.get_dummies(dfo['gind'], prefix='gind')[col]

#%%

def create_finaldfind(dfo0x, year_list):
    NIMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'NIMTA']]
    TLMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'TLMTA']]
    CASHMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'CASHMTA']]
    MB = dfo0x.loc[:, ['PERMCO', 'fyear', 'MB']]
    SIGMA = dfo0x.loc[:, ['PERMCO', 'fyear', 'SIGMA']]
    RSIZE = dfo0x.loc[:, ['PERMCO', 'fyear', 'RSIZE']]
    EXRET = dfo0x.loc[:, ['PERMCO', 'fyear', 'EXRET']]
    PRICE = dfo0x.loc[:, ['PERMCO', 'fyear', 'PRICE']]
    gind_15 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_15']]
    gind_20 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_20']]
    gind_25 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_25']]
    gind_30 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_30']]
    gind_35 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_35']]
    gind_40 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_40']]
    gind_45 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_45']]
    gind_50 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_50']]
    gind_55 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_55']]
    gind_60 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_60']]
    
    varstr = ['NIMTA', 'TLMTA', 'CASHMTA', 'MB', 'SIGMA', 'RSIZE', 'EXRET'
              , 'PRICE', 'gind_15', 'gind_20', 'gind_25', 'gind_30',
              'gind_35', 'gind_40', 'gind_45', 'gind_50', 'gind_55', 'gind_60']
    var = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]

    varcols = {c: [] for c in varstr}
    
    for i in list(range(18)):
        var[i] = var[i].set_index(['PERMCO','fyear'])[varstr[i]].unstack()
        for y in year_list:
            varcols[varstr[i]].append('{}_{}'.format(varstr[i], y))
        var[i].columns = varcols[varstr[i]]
    
    var_0x1 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]
    var_0x2 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]
    var_0x3 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]
    var_0x4 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]
    var_yrs = [var_0x1,var_0x2,var_0x3,var_0x4]
    
    for j in list(range(4)):
        for i in list(range(18)):
            var_yrs[j][i] = var[i].iloc[:,[j,j+1]]
            
    df_yrs = [var_yrs[0][0], var_yrs[1][0], var_yrs[2][0], var_yrs[3][0]]
    
    for j in list(range(4)):
        for df_ in var_yrs[j][1:]:
            df_yrs[j] = df_yrs[j].merge(df_, on='PERMCO')
    
    ind_todrop = [16,18,20,22,24,26,28,30,32,34]
    for j in list(range(4)):
        df_yrs[j].drop(columns=df_yrs[j].columns[ind_todrop], inplace=True)
    
    col_names = ['NIMTA_t','NIMTA_t+1','TLMTA_t','TLMTA_t+1','CASHMTA_t','CASHMTA_t+1','MB_t','MB_t+1','SIGMA_t',
             'SIGMA_t+1','RSIZE_t','RSIZE_t+1','EXRET_t','EXRET_t+1','PRICE_t','PRICE_t+1',
              'gind_15', 'gind_20', 'gind_25', 'gind_30',
              'gind_35', 'gind_40', 'gind_45', 'gind_50', 'gind_55', 'gind_60']
    
    for j in list(range(4)):
        df_yrs[j].columns = col_names
        df_yrs[j]['PERMCO_Y'] = df_yrs[j].index.astype(str) + '_' + str(year_list[j]+2)
        df_yrs[j].set_index('PERMCO_Y', inplace=True)
        
    df_yrs[0] = df_yrs[0].append(df_yrs[1])
    df_yrs[0] = df_yrs[0].append(df_yrs[2])
    df_yrs[0] = df_yrs[0].sort_index()
    
    return df_yrs[0]

#%%

df_ind = [pd.DataFrame() for i in range(15)]
for i in list(range(15)):
    df_ind[i] = create_finaldfind(ogdf_list[i], year_lists[i])
    print(i)
    
#%%

for i in list(range(15)):
    df_ind[i] = append_dlrsn(ogdf_list[i], df_ind[i], year_lists[i])
    df_ind[i] = df_ind[i].sort_index()
    

#%%

# TEST

#%%

def create_finaldftestind(dfo0x, year_list):
    NIMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'NIMTA']]
    TLMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'TLMTA']]
    CASHMTA = dfo0x.loc[:, ['PERMCO', 'fyear', 'CASHMTA']]
    MB = dfo0x.loc[:, ['PERMCO', 'fyear', 'MB']]
    SIGMA = dfo0x.loc[:, ['PERMCO', 'fyear', 'SIGMA']]
    RSIZE = dfo0x.loc[:, ['PERMCO', 'fyear', 'RSIZE']]
    EXRET = dfo0x.loc[:, ['PERMCO', 'fyear', 'EXRET']]
    PRICE = dfo0x.loc[:, ['PERMCO', 'fyear', 'PRICE']]
    gind_15 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_15']]
    gind_20 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_20']]
    gind_25 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_25']]
    gind_30 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_30']]
    gind_35 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_35']]
    gind_40 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_40']]
    gind_45 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_45']]
    gind_50 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_50']]
    gind_55 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_55']]
    gind_60 = dfo0x.loc[:, ['PERMCO', 'fyear', 'gind_60']]
    
    varstr = ['NIMTA', 'TLMTA', 'CASHMTA', 'MB', 'SIGMA', 'RSIZE', 'EXRET'
              , 'PRICE', 'gind_15', 'gind_20', 'gind_25', 'gind_30',
              'gind_35', 'gind_40', 'gind_45', 'gind_50', 'gind_55', 'gind_60']
    var = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]

    varcols = {c: [] for c in varstr}
    
    for i in list(range(18)):
        var[i] = var[i].set_index(['PERMCO','fyear'])[varstr[i]].unstack()
        for y in year_list:
            varcols[varstr[i]].append('{}_{}'.format(varstr[i], y))
        var[i].columns = varcols[varstr[i]]
    
    var_0x1 = [NIMTA, TLMTA, CASHMTA, MB, SIGMA, RSIZE, EXRET, PRICE, 
           gind_15, gind_20, gind_25, gind_30, gind_35, gind_40, gind_45, 
           gind_50, gind_55, gind_60]

    for i in list(range(18)):
        var_0x1[i] = var[i].iloc[:,[0,1]]
     
    df_vars = var_0x1[0]
    
    for df_ in var_0x1[1:]:
        df_vars = df_vars.merge(df_, on='PERMCO')
        
    ind_todrop = [16,18,20,22,24,26,28,30,32,34]
    df_vars.drop(columns=df_vars.columns[ind_todrop], inplace=True)
    
    col_names = ['NIMTA_t','NIMTA_t+1','TLMTA_t','TLMTA_t+1','CASHMTA_t','CASHMTA_t+1','MB_t','MB_t+1','SIGMA_t',
             'SIGMA_t+1','RSIZE_t','RSIZE_t+1','EXRET_t','EXRET_t+1','PRICE_t','PRICE_t+1',
              'gind_15', 'gind_20', 'gind_25', 'gind_30',
              'gind_35', 'gind_40', 'gind_45', 'gind_50', 'gind_55', 'gind_60']
    
    df_vars.columns = col_names
    df_vars['PERMCO_Y'] = df_vars.index.astype(str) + '_' + str(year_list[0]+2)
    df_vars.set_index('PERMCO_Y', inplace=True)
        
    df_vars = df_vars.sort_index()
    
    return df_vars

#%%

df_tind = [pd.DataFrame() for i in range(15)]
for i in list(range(15)):
    df_tind[i] = create_finaldftestind(ogdf_test[i], year_lists[i][3:5])
    
#%%

for i in list(range(15)):
    df_tind[i] = append_dlrsntest(ogdf_test[i], df_tind[i], year_lists[i])
    df_tind[i] = df_tind[i].sort_index()
    
#%%

for i in list(range(15)):
   df_ind[i].to_csv('dftrain' + str(i) + 'ind.csv')

#%%

for i in list(range(15)): 
   df_tind[i].to_csv('dftest' + str(i) + 'ind.csv')