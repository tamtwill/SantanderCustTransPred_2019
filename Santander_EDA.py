#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:54:36 2019

https://www.kaggle.com/c/santander-customer-transaction-prediction
April 10, 2019 - Final submission deadline

@author: Tamara Williams
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


## see  https://github.com/numpy/numpy/issues/11411 and 
## https://stackoverflow.com/questions/53334421/futurewarning-with-distplot-in-seaborn 
## for why FutureWarnigns are suppressed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

# set a pretty default palette
sns.palplot(sns.husl_palette(10, h=.5))
col_list_palette = sns.husl_palette(10, h=.5)
sns.set_palette(col_list_palette)



datapath = '~/gitRepo/SantanderCustTransPredictions/data/'
outpath = '~/gitRepo/SantanderCustTransPredictions/output/'

train = pd.read_csv(datapath+'train.csv', delimiter=',', encoding='utf-8')

# look at general aspects of the data
print(train.shape,'\n')
print(train.head(5),'\n')
print(train.info(), '\n')
print(train.dtypes, '\n')

print(train.describe(),'\n')

# explicitly check for missing values
train.isnull().values.any()
train.isna().sum()


print (train.target.value_counts())
print('Transaction rate = ', 100*(train.target.value_counts()[1]/train.target.value_counts()[0]))

t_data = train.drop(columns = ['ID_code'], axis = 1)
    
# now the numeric data, ints first    
plot_number = 1
plt.figure(figsize=(12, 16), facecolor='white')
df_ints = t_data.select_dtypes([int])
for c in df_ints.columns:
    ax = plt.subplot(8, 3, plot_number)
    sns.distplot(train[c])
    plt.xticks(rotation=45)
    plot_number = plot_number+1
plt.tight_layout()
plt.show()
       
plot_number = 1
plt.figure(figsize=(12, 288), facecolor='white')
df_ints = t_data.select_dtypes([float])
for c in df_ints.columns:
    ax = plt.subplot(200, 4, plot_number)
    sns.distplot(train[c])
    plt.xticks(rotation=45)
    plot_number = plot_number+1
plt.tight_layout()
plt.show()
    

## looking at a correlation matrix
t_corr = t_data.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(t_corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(t_corr, mask=mask, vmax=.3, square=True, center = 0, cmap=sns.color_palette("BrBG", 7))
plt.show()

print(t_corr)


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

t_data = t_data.drop(['target'], axis = 1)
X = add_constant(t_data)
VIF = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
VIF.columns = 'VIF_val'
print(VIF)
VIF_review = VIF[VIF >= 5]
