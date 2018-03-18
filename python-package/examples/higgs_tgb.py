import tgboost as tgb

# training phase
ftrain = "~/PycharmProjects/data/train_.csv"
fval = "~/PycharmProjects/data/test_.csv"
params = {'categorical_features': ["PRI_jet_num"],
          'early_stopping_rounds': 10,
          'maximize': True,
          'eval_metric': 'auc',
          'loss': 'logloss',
          'eta': 0.3,
          'num_boost_round': 20,
          'max_depth': 7,
          'scale_pos_weight':1.,
          'subsample': 0.8,
          'colsample': 0.8,
          'min_child_weight': 1.,
          'min_sample_split': 5,
          'reg_lambda': 1.,
          'gamma': 0.,
          'num_thread': -1
          }

model = tgb.train(ftrain, fval, params)

# testing phase
ftest = "~/PycharmProjects/data/test_.csv"
foutput = "~/PycharmProjects/data/test_preds.csv"
model.predict(ftest, foutput)
