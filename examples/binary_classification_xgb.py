import xgboost as xgb
import pandas as pd

# data set : https://pan.baidu.com/s/1c23gJkc?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=
train = pd.read_csv('../../train.csv')
val = pd.read_csv('../../test.csv')

train_y = train.label
train_X = train.drop('label', axis=1)
val_y = val.label
val_X = val.drop('label', axis=1)

dtrain = xgb.DMatrix(train_X, label=train_y)
dval = xgb.DMatrix(val_X, label=val_y)

params = {'booster':'gbtree',
          'objective': 'binary:logistic',
          'eta': 0.3,
          'max_depth': 7,
          'num_boost_round': 10000,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 1.0,
          'min_sample_split': 50,
          'min_child_weight': 1,
          'lambda': 1,
          'gamma': 0.01,
          'eval_metric': "error",
          'maximize': False}

watchlist = [(dval, 'val')]
model = xgb.train(params, dtrain,num_boost_round=10000, early_stopping_rounds=20, evals=watchlist)
#  Best iteration 56, val-error:0.201
