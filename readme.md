
##What is TGBoost
It is a **T**iny implement of **G**radient **Boost**ing tree, according to the xgboost algorithm, and support some features in xgboost:

- Built-in loss and Customized loss

	- Built-in loss contains: Square error loss for regression task, Logistic loss for classification task	
	- For customize loss function,use `autograd` to calculate the grad and hess automaticly

- Multi-processing 

	when finding best tree node split
	
- Feature importance
- Regularization

	lambda, gamma (as in xgboost scoring function)

- Randomness
	- rowsample
	- colsample_bytree
	- colsample_bylevel

##Dependence
TGBoost use `Pandas.DataFrame` to store data, and `autograd` to take derivation.

- Pandas
- Numpy
- autograd

##Example

Here is a classification task example:

```python

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
```

You can also define your own loss function:

```python

from tgboost import TGBoost
import pandas as pd
import autograd.numpy as anp

train = pd.read_csv('train.csv')
train_y = train.label
train_X = train.drop('label', axis=1)

def logistic_loss(pred,y):
    return -(y*anp.log(pred) + (1-y)*anp.log(1-pred))

params = {'loss': logistic_loss,
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

```

## TODO
- reduce memory usage
- speed up training and predicting
- sample weights

   	this is easy to implement, change `grad` and `hess` to:  `weight*grad`,  `weight*hess`

- post prunning

	the current implement just stop growing the tree when the split gain is negative, better to use post prunning instead(as in xgboost)

- metric: 
	
	auc, mse, mae, precision, recall, f_score, etc.

- early stopping
- cross validation



##Reference

- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Higgs Boson Discovery with Boosted Trees](http://www.jmlr.org/proceedings/papers/v42/chen14.pdf)
