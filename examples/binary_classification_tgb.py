from tgboost import tgb
import pandas as pd

# data set : https://pan.baidu.com/s/1c23gJkc?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=
train = pd.read_csv('../../train.csv')
val = pd.read_csv('../../test.csv')

train_y = train.label.values
train_X = train.drop('label', axis=1).values
val_y = val.label.values
val_X = val.drop('label', axis=1).values

print train_X.shape, val_X.shape

params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 6,
          'num_boost_round': 200,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample': 0.8,
          'min_sample_split': 50,
          'min_child_weight': 1,
          'reg_lambda': 1,
          'gamma': 0.1,
          'eval_metric': "error",
          'early_stopping_rounds': 20,
          'maximize': False}

tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)
# TGBoost training Stop, best round is 194, best val-error is 0.204
