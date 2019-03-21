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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from xgboost.sklearn import XGBRegressor as sk_xgb


# Global Variables
# seed value for random number generators to obtain reproducible results
RANDOM_SEED = 31
RANDOM_SEED_MODEL = 3131
CUTOFF = .0015



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
print ('\n')

num_var = cumm_var.shape[0]

for i in range(num_var):
    if cumm_var[i] < 98.00:
        print ("PC %i accounts for %g%% of variation; cummulative variation is: %g%%"\
        %(i+1, var_ex[i], cumm_var[i]))
        last_i = i

print ('\n\n')
print("------------------------------------------------")
print ("Time to fit/find Principle Components: %.5f" %pca_time)

# plot the results on a scree plot
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Scree Plot', fontsize=14)
plt.ylabel('% of Variance Explained')
plt.xlabel('# of Principal Components')
plt.plot(cumm_var, label = "% Cummulative Explained Variance")
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

# compute full set of principal components (scores)
pca_scores = pca.fit_transform(X)
print(pca_scores)
print(pca.components_)

# show per PCA column weights
print(pd.DataFrame(pca.components_,columns=t_data.columns))

# show per PCA column weights
print(pca.explained_variance_ratio_)

# Recursive Feature Elimination
model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=6)
rfe = RFE(estimator=model, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_
features = t_data.columns
tmp = np.vstack((features, ranking)).T
select_features = pd.DataFrame(tmp)
select_features.columns = ('Feature', 'Rank')
select_features = select_features.sort_values(by='Rank', inplace = False)
print(select_features)


# look at XGBoost feature importance
bst = sk_xgb(max_depth=6, 
         learning_rate=0.5, 
         n_estimators= 50, 
         objective='reg:linear',         
         booster='gbtree',
         n_jobs=6, 
         random_state=RANDOM_SEED,
         eval_metric='auc',
         silent = False)
bst.fit(X,y)

# Nice graph
fig, ax = plt.subplots(figsize=(10, 80))
sk_xgb.plot_importance(bst, ax = ax)

# make a function
features = t_data.columns
tmp = np.vstack((features,bst.feature_importances_)).T
select_features = pd.DataFrame(tmp)
select_features.columns = ('Feature', 'Importance')
select_features = select_features.sort_values(by='Importance', ascending = False, inplace = False)
print(select_features)


# make a function
# make new smaller dataset
selected_features = select_features.loc[select_features['Importance'] > CUTOFF]
select_list = selected_features['Feature']
select_train_list = select_list.append(pd.Series(['target', 'ID_code']), ignore_index=True)
select_test_list = select_list.append(pd.Series(['ID_code']), ignore_index=True)

small_train = train[select_train_list]
small_train.to_csv(datapath+'small_train2'+'.csv', sep = ',', index = False, encoding = 'utf-8')

# now do the test data too
test = pd.read_csv(datapath+'test.csv', delimiter=',',  encoding='utf-8-sig')
small_test = test[select_test_list]
small_test.to_csv(datapath+'small_test2'+'.csv', sep = ',', index = False, encoding = 'utf-8')

