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
import statsmodels.api as sm
from scipy import stats


from sklearn.linear_model import LinearRegression
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

# try a standard regression models, did OK in eval
model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=6).fit(X, y)
my_pred1 = model.predict(tst_data)
make_submission(my_pred1, 'LinearRegression1_')

model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=6).fit(X, y).fit(X, y)
my_pred2 = model.predict(tst_data)
make_submission(my_pred2, 'LinearRegression2_')

# compare model packages
model = sm.OLS(y,X).fit()
my_predOLS1 = model.predict(tst_data)
make_submission(my_predOLS1, 'OLS1_')


