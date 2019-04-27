import numpy as np 
import time
import pandas as pd 
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')


datapath = '~/gitRepo/SantanderCustTransPredictions/data/'
outpath = '~/gitRepo/SantanderCustTransPredictions/output/'


def make_submission(predictions, model):
    pred = pd.DataFrame(predictions)
    submission = pd.concat([test_index.reset_index(drop=True), pred], axis=1)
    submission.columns=['ID_code', 'target']
    dt = time.strftime('%Y-%m-%d %H.%M.%S', time.localtime())
    filename = outpath+"tw_"+str(model)+dt+'.csv'
    submission.to_csv(filename, sep = ',', date_format = 'string', index = False, encoding = 'utf-8')



def train_model(params):
    num_round = 100000
    folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=31)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    
    for fold_, (train_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("Fold {}".format(fold_))
        train_data = lgb.Dataset(train.iloc[train_idx][features], label=target.iloc[train_idx])
        val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])
        bst = lgb.train(param, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=1000, early_stopping_rounds = 2500)
        oof[val_idx] = bst.predict(train.iloc[val_idx][features], num_iteration=bst.best_iteration)
        predictions += bst.predict(test[features], num_iteration=bst.best_iteration) / folds.n_splits
    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

    make_submission(predictions, 'lgbm_1_')



# get the data
#train = pd.read_csv('~/gitRepo/SantanderCustTransPredictions/data/train.csv')
#test = pd.read_csv('~/gitRepo/SantanderCustTransPredictions/data/test.csv')

train = pd.read_csv('~/gitRepo/SantanderCustTransPredictions/data/small_train.csv')
test = pd.read_csv('~/gitRepo/SantanderCustTransPredictions/data/small_test.csv')

test.shape, train.shape

features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']

test_index = test['ID_code']
tst_data = test.drop(columns=['ID_code'])




###********** run this once
#print("Parameter optimization")
#tmp = train.drop(['ID_code', 'target'], axis = 1)
#t_data = lgb.Dataset(tmp, label=target)
#lgb_model =  lgb.LGBMRegressor()
#model = GridSearchCV(lgb_model,
#    {
#    'bagging_freq': [1, 3, 5],
#    'bagging_fraction': [.25, .5, 1],
#    'boost': ['gbdt','gbrt'],
#    'feature_fraction': [0.25, .5, .75, 1],
#    'learning_rate': [0.015, .02, .1],
#    'min_data_in_leaf': [20, 25, 50],
#    'num_leaves': [10, 15, 20,100],
#    'tree_learner': ['serial', 'feature'],
#    'metric':['auc'],
#    'is_unbalance':[ True]},
#    n_jobs=8,
#    verbose=1
#)

## best values from gridsearch 1
# tmp = train.drop(['ID_code', 'target'], axis = 1)
# t_data = lgb.Dataset(tmp, label=target)
# lgb_model =  lgb.LGBMRegressor()
#param = {
#     'bagging_fraction': 0.5,
#      'bagging_freq': 3,
#      'boost': 'gbdt',
#      'feature_fraction': 1,
#      'is_unbalance': True,
#      'learning_rate': 0.1,
#      'metric': 'auc',
#      'min_data_in_leaf': 50,
#      'num_leaves': 100,
#      'tree_learner': 'serial'}   
#
#train_model(param)

## old values
# param = {
#     'bagging_freq': 5,
#     'bagging_fraction': 0.33,
#     'boost_from_average':'false',
#     'boost': 'gbdt',
#     'feature_fraction': 0.05,
#     'learning_rate': 0.015,
#     'max_depth': -1,
#     'metric':'auc',
#     'min_data_in_leaf': 100,
#     'min_sum_hessian_in_leaf': 10.0,
#     'num_leaves': 15,
#     'num_threads': 20,
#     'tree_learner': 'serial',
#     'objective': 'binary',
#     'verbosity': 1
# }
# train_model(param)


##********** refining the values
print("Parameter optimization round 2")
tmp = train.drop(['ID_code', 'target'], axis = 1)
t_data = lgb.Dataset(tmp, label=target)
lgb_model =  lgb.LGBMRegressor()

model = GridSearchCV(lgb_model,
{'bagging_fraction': [0.5,.6],
'bagging_freq': [3,4],
'boost': 'gbdt',
'feature_fraction': [1, 1.5],
'is_unbalance': [True, False],
'learning_rate': [0.1,.2],
'metric': 'auc',
'min_data_in_leaf': [50, 100, 150],
'num_leaves': [100, 150, 200],
'tree_learner': 'serial'})

model.fit(tmp, target)

model.best_score_
print(model.best_score_)
print(model.best_params_)
model.best_params_


train_model(param)

num_round = 100000
folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=31)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
   print("Fold {}".format(fold_))
   trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
   val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])
   model = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 2500)
   oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
   predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

submission = pd.DataFrame({"ID_code": test.ID_code.values})
submission["target"] = predictions
submission.to_csv('~/gitRepo/SantanderCustTransPredictions/output/lgbm_magic.csv', index=False)





