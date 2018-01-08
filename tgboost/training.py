from .gbm import TGBoost
import numpy as np

def train(features,
          label,
          validation_data=(None, None),
          early_stopping_rounds=np.inf,
          maximize=True,
          eval_metric=None,
          loss="logisticloss",
          eta=0.3,
          num_boost_round=1000,
          max_depth=6,
          scale_pos_weight=1,
          subsample=0.8,
          colsample=0.8,
          min_child_weight=1,
          min_sample_split=10,
          reg_lambda=1.0,
          gamma=0,
          num_thread=-1):

          tgb = TGBoost()
          tgb.fit(features,label,validation_data,early_stopping_rounds,maximize,eval_metric,
                  loss,eta,num_boost_round,max_depth,scale_pos_weight,subsample,colsample,
                  min_child_weight,min_sample_split,reg_lambda,gamma,num_thread)
          return tgb
