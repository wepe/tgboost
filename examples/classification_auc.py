from tgboost import TGBoost
import pandas as pd

train = pd.read_csv('train.csv')
train = train.sample(frac=1.0, axis=0)  # shuffle the data
val = train.iloc[0:5000]
train = train.iloc[5000:]


train_y = train.label
train_X = train.drop('label', axis=1)
val_y = val.label
val_X = val.drop('label', axis=1)


params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 6,
          'num_boost_round': 500,
          'scale_pos_weight': 1.0,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 1.0,
          'min_sample_split': 10,
          'min_child_weight': 2,
          'reg_lambda': 10,
          'gamma': 0,
          'eval_metric': "auc",
          'early_stopping_rounds': 20,
          'maximize': False,
          'num_thread': 16}


tgb = TGBoost()
tgb.fit(train_X, train_y, validation_data=(val_X, val_y), **params)
