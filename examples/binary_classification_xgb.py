import xgboost as xgb
import pandas as pd

train = pd.read_csv('../data/train.csv')
train = train.sample(frac=1.0, axis=0)  # shuffle the data
val = train.iloc[0:5000]
train = train.iloc[5000:]

train_y = train.label
train_X = train.drop('label', axis=1)
val_y = val.label
val_X = val.drop('label', axis=1)

dtrain = xgb.DMatrix(train_X, label=train_y)
dval = xgb.DMatrix(val_X, label=val_y)

params = {'booster':'gbtree',
          'objective': 'binary:logistic',
          'eta': 0.3,
          'max_depth': 6,
          'num_boost_round': 10000,
          'scale_pos_weight': 1.0,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 1.0,
          'min_sample_split': 10,
          'min_child_weight': 2,
          'lambda': 10,
          'gamma': 0,
          'eval_metric': "error",
          'maximize': False,
          'num_thread': 16}

watchlist = [(dval, 'val')]
model = xgb.train(params, dtrain,num_boost_round=10000, early_stopping_rounds=20, evals=watchlist)
#  Best iteration 37, val-error:0.198
