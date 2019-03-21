#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:18:12 2019

@author: Tamara Williams

N.B.: After running this, it look like GradientBoosting and possibly Linear models are 
worth developing further.

"""


import pandas as pd
import numpy as np

#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,\
    GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,\
    roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.exceptions import ConvergenceWarning


from matplotlib import pyplot as plt


RND_ST = np.random.RandomState(31)

# initialize various system variables
RANDOM_SEED = 13
SET_FIT_INTERCEPT=True

# set the number of folds for cross-validation
#N_FOLDS = 10
N_FOLDS = 5

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='error', category=ConvergenceWarning) 


pd.set_option('display.max_columns', 10)

datapath = '~/gitRepo/SantanderCustTransPredictions/data/'
outpath = '~/gitRepo/SantanderCustTransPredictions/output/'


train = pd.read_csv(datapath+'train.csv', delimiter=',', encoding='utf-8')
t_data = train.copy(deep=True)
t_data = t_data.drop(columns=['ID_code','target'])
target = train['target']


# Build and cross-validate regression models
#--------------------------------------------------
# Setup the list of models to look at, let's try a range

mod_methods = ['LogisiticRegression', 'RidgeClassifier', 'BaggingClassifier', 
          'RandomForest', 'GradientBoosting 1.0','GradientBoosting .1', 
          'Extra Trees', 'BernoulliNB']  #'AdaBoost',

mod_list = [LogisticRegression(fit_intercept = SET_FIT_INTERCEPT),                
               RidgeClassifier(alpha = 1, solver = 'auto', 
                     fit_intercept = SET_FIT_INTERCEPT, 
                     normalize = False, 
                     random_state = RANDOM_SEED),
               BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_SEED, max_features='log2'), 
                    n_estimators=100,max_samples=100, bootstrap=True, 
                    n_jobs=-1, random_state=RANDOM_SEED),
               RandomForestClassifier(n_estimators=100, max_leaf_nodes=12, bootstrap=True,
                    n_jobs=-1, random_state=RANDOM_SEED, max_features='log2'),
               GradientBoostingClassifier(max_depth=5, n_estimators=100, 
                    learning_rate=1.0, random_state=RANDOM_SEED, max_features='log2'),
               GradientBoostingClassifier(max_depth=5, n_estimators=100, 
                    learning_rate=0.1, random_state=RANDOM_SEED, max_features='log2'),
               ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=5, 
                    min_samples_split=2, min_samples_leaf=1, max_features='log2', 
                    bootstrap=True, random_state=RANDOM_SEED),
               BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
                ]
         
   


# let's evaulate using Strat Shuffle to preserve the ratios of the response
# variables in each fold of the test set
print ("\n\n\n************ Using Stratefied Shuffle Split ***********************")

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

def eval_model(data, labels, oversample):   

    # array to hold results
    cross_val_res1 = np.zeros((N_FOLDS, len(mod_methods)))
    cross_val_res2 = np.zeros((N_FOLDS, len(mod_methods)))
    r2_val_res = np.zeros((N_FOLDS, len(mod_methods)))
    auc_val_res = np.zeros((N_FOLDS, len(mod_methods)))
    
    # N -fold cross-validation with stratified sampling.
    stsp = StratifiedShuffleSplit(n_splits=N_FOLDS)
    fold_index = 0
    for train_index, test_index in stsp.split(data, labels):
       X_train, X_test = data.iloc[train_index], data.iloc[test_index]
       y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
       
       if oversample == True:
           sm = SMOTE(sampling_strategy='minority')
           X_train, y_train = sm.fit_sample(X_train, y_train)
       
       print ("\n-----------------------------------------------------------------------------------") 
       print ("--------------------------------------- FOLD = {} ----------------------------------".format (fold_index))
       print ("-----------------------------------------------------------------------------------\n")  
         
       method_index = 0
       for model_name, method in zip(mod_methods, mod_list):
           print("\n\n\nRegression model evaluation for model:", model_name)
           print("Scikit_Learn method:", method)
           try:
               method.fit(X_train,y_train)
           except ConvergenceWarning:
               print ('###########################################################################')  
               print ('########################### Failed to converge  ###########################')  
               print ('###########################################################################')  
               method_index += 1      
               continue
           y_test_predict = method.predict(X_test)
           r2_val = r2_score(y_test, y_test_predict) 
           print("R-squared is:", r2_val)
           fold_method_res1 = mean_absolute_error(y_test, y_test_predict)
           fold_method_res2 = np.sqrt(mean_squared_error(y_test, y_test_predict))
           fold_method_auc = roc_auc_score(y_test, y_test_predict)
           print(method.get_params(deep=True))
           print('Mean absolute error:', fold_method_res1)
           print('Root mean-squared error:', fold_method_res2)
           print("Area under the ROC :", fold_method_auc)

           cross_val_res1[fold_index, method_index] = fold_method_res1
           cross_val_res2[fold_index, method_index] = fold_method_res2
           r2_val_res[fold_index, method_index] = r2_val    
           auc_val_res[fold_index, method_index] = fold_method_auc
            
           fpr, tpr, _ = roc_curve(y_test, y_test_predict)
           plt.figure(figsize=(8, 6))
           plt.title("ROC Curve")
           plot_roc_curve(fpr, tpr)
           plt.show()
          
           method_index += 1

       fold_index += 1
       

    cross_val_res1_df = pd.DataFrame(cross_val_res1)
    cross_val_res1_df.columns = mod_methods
    
    cross_val_res2_df = pd.DataFrame(cross_val_res2)
    cross_val_res2_df.columns = mod_methods
    
    r2_val_res_df = pd.DataFrame(r2_val_res)
    r2_val_res_df.columns = mod_methods
    
    auc_val_res_df = pd.DataFrame(auc_val_res)
    auc_val_res_df.columns = mod_methods

    res1 = cross_val_res1_df.mean()
    res2 = cross_val_res2_df.mean()
    r2 = r2_val_res_df.mean()
    auc = auc_val_res_df.mean()
    
    tmp = pd.concat([res1, res2, r2, auc], axis=1)   

    return tmp


#**************************************************
# Run the build and evaluate model loop
#**************************************************
print ("\n\n\n**************************** REGULAR STRATEFIED RESULTS ****************************")
print ("******************************************************************************\n ")
print ("\n\n\n**************************** PER FOLD REGRESSION RESULTS ****************************\n\n")
orig_res = eval_model(t_data, target, False)
orig_res.columns = ['MAE', 'RMSE', 'R2', 'AUC']
sorted_res_1 = orig_res.sort_values(by = 'MAE')



########################################################################
# get the feature importance for the model that did best
########################################################################
def get_importance(df, model_name):  
          
    col_names = df.columns
    feature_list = np.delete(col_names,0)
         
    
    X_train = df.iloc[:, 1:]
    y_train = df.iloc[:, 0]
           
    try:
        #for mod_list[model]:
        model_num = mod_methods.index(model_name)
        model = mod_list[model_num]
        
        model.fit(X_train,y_train)
        feature_import = np.round(model.feature_importances_,4)
        array_stack = np.column_stack([feature_list, feature_import])
        tmp_array = array_stack[np.argsort(array_stack[:, 1])]
        print('\n----------------------------------------------')
        print('Feature importance for method', model, '\n')
        print(np.array2string(tmp_array).replace('[[',' [').replace(']]',']'))
    except:
        print("**** !! Best method has no feature importance  !! ****", candidate)
       

print ("\n\n\n************ FEATURE IMPORTANCE ***********************")

# get model with lowest MAE and find feature importance
candidate = sorted_res_1.index[0]
get_importance(train, candidate)


########################################################################
# OK, let's look at the impact of over-sampling the people who leave
# the R^2 is very low, and it would be good to see if there is something
# we can do to improve the model
########################################################################

orig_res = eval_model(t_data, target, True)
orig_res.columns = ['MAE', 'RMSE', 'R2', 'AUC']
sorted_res_2 = orig_res.sort_values(by = 'MAE')



# Output results of regular cross-validation for comparison
#--------------------------------------------------
print ("\n\n\n************************** REGULAR SAMPLING RESULTS ***************************")
print ("******************************************************************************\n ")
print('Average results from ', N_FOLDS, '-fold cross-validation\n', sep = '')     
print("Method\n{0}".format(sorted_res_1))


# Output results of oversampled cross-validation for comparison
#--------------------------------------------------
print ("\n\n\n**************************** OVERSAMPLED RESULTS ****************************")
print ("******************************************************************************\n ")
print ("\n************ AVERAGE OF REGRESSION RESULTS ACROSS ALL FOLDS ")
print('Average results from ', N_FOLDS, '-fold cross-validation\n', sep = '')     
print("Method\n{0}".format(sorted_res_2))