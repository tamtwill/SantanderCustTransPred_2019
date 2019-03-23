#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:09:05 2019

@author: Tamara Williams
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Global Variables
# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 31
RANDOM_SEED_MODEL = 31



# get the dataset 
#----------------------------------------------------------
pd.set_option('display.max_columns', 10)

datapath = '~/gitRepo/SantanderCustTransPredictions/data/'
outpath = '~/gitRepo/SantanderCustTransPredictions/output/'

# read the data and prep the train/test sets
train = pd.read_csv(datapath+'train.csv', delimiter=',',  encoding='utf-8-sig')

t_data = train.drop(columns = ['ID_code','target']) 
y = np.array(train['target'])
X = t_data.values

num_var = train.shape[1]

pca = PCA()  

# start the timing  for PCA components
start_time = time.time()
pca.fit_transform(X)
# end timing
end_time = time.time()
pca_time = end_time - start_time

##amount of variance each PC explains an the cummulative variance explained
var_ex = np.round(pca.explained_variance_ratio_, decimals = 3)*100
cumm_var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 3)*100)
eg_val = np.round(pca.explained_variance_)
print ('\n')

pc_lst = []
for i in range(len(var_ex)):
    if cumm_var[i] < 100.00:
        print ("PC %i accounts for %g%% of variation; cummulative variation is: %g%%"\
        %(i+1, var_ex[i], cumm_var[i]))
        last_i = i
        pc_lst.append(i) 

print ('\n\n')
print("------------------------------------------------")
print ("Time to fit/find Principle Components: %.5f" %pca_time)


# plot the results on a scree plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Scree Plot', fontsize=14)
plt.ylabel('Eigenvalues')
plt.xlabel('Principal Components')
plt.plot(eg_val, label = "Eigenvalues")
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(1, num_var, 50.0))  
plt.xticks(rotation=90)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.savefig("pca_sree_plot", 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()

# or, if you prefer, look at the explained cummulative variance
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Explained Variance', fontsize=14)
plt.ylabel('% of Variance Explained')
plt.xlabel('# of Principal Components')
plt.plot(cumm_var, label = "% Cummulative Explained Variance")
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(1, num_var, 50.0))  
plt.xticks(rotation=90)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.savefig("pca_exp_var_plot", 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()


# bar chart version, a bit easy to see the jumps in variance contribution
fig = plt.figure(figsize = (32,8))
plt.title('Explained Variance', loc = 'left', fontsize=14)
plt.ylabel('% of Variance Explained')
plt.xlabel('# of Principal Components')
bar_df = pd.DataFrame({'var':pca.explained_variance_ratio_,'PC': pc_lst})
sns.barplot(x='PC',y="var", data=bar_df, color="c")
plt.savefig("pca_exp_var_barplot", 
    bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 
    orientation='portrait', papertype=None, format=None, 
    transparent=True, pad_inches=0.25, frameon=None)  
plt.show()



# compute full set of principal components (scores)
pca_scores = pca.fit_transform(X)