#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:20:02 2018

@author: Tamara Williams
"""
import time
import pandas as pd
import numpy as np
import xgboost as xgb

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import auc, roc_auc_score





pd.set_option('display.max_columns', 10)

datapath = '~/gitRepo/SantanderCustTransPredictions/data/'
outpath = '~/gitRepo/SantanderCustTransPredictions/output/'

dataset = ''
#dataset = 'small_'



# initialize various system variables
#RND_ST = np.random.RandomState(31)

RANDOM_SEED = 31
SET_FIT_INTERCEPT=True

# set the number of folds for cross-validation
#N_FOLDS = 10
N_FOLDS = 5
#N_ROUNDS = 500
N_ROUNDS = 50

def make_submission(predictions, model):
    pred = pd.DataFrame(predictions)
    submission = pd.concat([test_index.reset_index(drop=True), pred], axis=1)
    submission.columns=['ID_code', 'target']
    dt = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
    filename = outpath+"tw_"+str(model)+dt+'_'+dataset+'.csv'
    submission.to_csv(filename, sep = ',', date_format = 'string', index = False, encoding = 'utf-8')

# read the data and prep the train/test sets
train = pd.read_csv(datapath+dataset+'train.csv', delimiter=',',  encoding='utf-8-sig')
test = pd.read_csv(datapath+dataset+'test.csv', delimiter=',',  encoding='utf-8-sig')

test_index = test['ID_code']
tst_data = test.drop(columns=['ID_code'])

t_data = train.drop(columns = ['ID_code','target']) 
y = np.array(train['target'])
X = t_data.values


# make an XGBoost model to see how it does
dtrain = xgb.DMatrix(X, label = train['target'])

num_round = N_ROUNDS
param = {'max_depth':6, 
         'learning_rate':0.1, 
         'n_estimators':500, 
         'objective':'reg:linear',         
         'booster':'gbtree',
         'n_jobs':8, 
         'subsample':1, 
         'colsample_bytree':1, 
         'colsample_bylevel':1,
         'seed':RANDOM_SEED,
         'eval_metric':'auc'}

bst = xgb.train(param, dtrain, num_round)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dtest = xgb.DMatrix(tst_data)
# do this to work around an XGB.DMatrix bug that is failing to convert col names 
# on test data.  See https://github.com/dmlc/xgboost/issues/1238
col_labels = dtrain.feature_names
dtest.feature_names = col_labels

my_pred = bst.predict(dtest)
make_submission(my_pred, 'XGB_lin')

# explore the most important features
#---------------------------------------------------------------
# Nice graph
fig, ax = plt.subplots(figsize=(10, 80))
xgb.plot_importance(bst, ax = ax)

import graphviz # need this to get tree
fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(bst, num_trees=4, ax=ax)
plt.show()

#
##!!!!!!!!!!!!!!!!!!!!!!!!!!!
## refine the hyperparameters to use
## RUN ONLY ONCE
##!!!!!!!!!!!!!!!!!!!!!!!!!!!
#model_xgb = xgb.XGBRegressor(objective = 'binary:logistic')
#
#param_dist = {'n_estimators': stats.randint(400, 800),
#              'learning_rate': stats.uniform(0.05, 0.1),
#              'subsample': stats.uniform(0.3, 0.7),
#              'max_depth': [5, 6, 7, 8, 9],
#              'colsample_bytree': stats.uniform(0.5, 0.45),
#              'min_child_weight': [1, 2, 3],
#
#             }
#
#model = RandomizedSearchCV(model_xgb, param_distributions = param_dist, 
#        n_iter = 25, cv = 5, scoring = 'roc_auc', n_jobs=8,verbose=1)
#
#model.fit(X,y)
#model.best_score_
#print(model.best_score_)
#print(model.best_params_)
#model.best_params_
#
#numFolds = 5
#folds = KFold(shuffle = True, n_folds = N_FOLDS)
#
#estimators = []
#results = np.zeros(len(X))
#score = 0.0
#for train_index, test_index in folds:
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    model_params.fit(X_train, y_train)
#
#    estimators.append(model_params.best_estimator_)
#    results[test_index] = model_params.predict(X_test)
#    score += f1_score(y_test, results[test_index])
#score /= numFolds
##!!!!!!!!!!!!!!!!!!!!!!!!!!!

##!!!!!!!!!!!!!!!!!!!!!!!!!!!
## refine the hyperparameters to use
## GridSeach version
##
## RUN ONLY ONCE
##**********
#print("Parameter optimization")
#xgb_model = xgb.XGBRegressor()
#model = GridSearchCV(xgb_model,
#                   {'max_depth': [6],
#                    'n_estimators': [500, 1000, 1500, 2000],
#                    'learning_rate':[.1]}, 
#                    n_jobs=8,
#                    verbose=1)
#model.fit(X,y)
#model.best_score_
#print(model.best_score_)
#print(model.best_params_)
#model.best_params_
