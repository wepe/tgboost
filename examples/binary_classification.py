from tgboost import TGBoost
import pandas as pd

train = pd.read_csv('train.csv')
train_y = train.label
train_X = train.drop('label', axis=1)

params = {'loss': "logisticloss",
          'eta': 0.3,
          'max_depth': 6,
          'num_boost_round': 100,
          'rowsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'min_sample_split': 10,
          'l2_regularization': 10,
          'gamma': 0.4,
          'eval_metric': "acc",
          'num_thread': 16}

tgb = TGBoost()
tgb.fit(train_X, train_y, **params)

preds = tgb.predict(data_X)
feature_importance = tgb.feature_importance
