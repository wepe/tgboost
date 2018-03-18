## What is TGBoost

It is a **T**iny implement of **G**radient **Boost**ing tree, based on  XGBoost's scoring function and SLIQ's efficient tree building algorithm. TGBoost build the tree in a level-wise way as in SLIQ (by constructing Attribute list and Class list). Currently, TGBoost support  parallel learning on single machine,  the speed and memory consumption are comparable to XGBoost.


TGBoost supports most features as other library:  

- **Built-in loss** , Square error loss for regression task, Logistic loss for classification task

- **Early stopping** , evaluate on validation set and conduct early stopping

-  **Feature importance** , output the feature importance after training

- **Regularization** , lambda, gamma

- **Randomness**, subsample，colsample

- **Weighted loss function** , assign weight to each sample


Another two features  are novel: 

- **Handle missing value**, XGBoost learn a direction for those with missing value, the direction is left or right. TGBoost take a different approach: it enumerate missing value go to left child, right child and missing value child, then choose the best one. So TGBoost use Ternary Tree.

-  **Handle categorical feature**, TGBoost order the categorical feature by their statistic (Gradient_sum / Hessian_sum) on each tree node, then conduct split finding as numeric feature.


## Installation

The current version is implemented in pure Java, to use TGBoost you should first install [JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html). For Python user, Python binding is also provided:

```
git clone git@github.com:wepe/tgboost.git
cd python-package
sudo python setup.py install
```

## To Understand TGBoost

For those want to understand how TGBoost work, and dive into Gradient Boosting Machine, please refer to the Python implementation of TGBoost: [tgboost-python](https://github.com/wepe/tgboost/tree/tgboost-python), the python source code is relatively easy to follow. 


## Example

Here is an example, download the data [here](https://pan.baidu.com/s/1dGDr7pR)

```python

import tgboost as tgb

# training phase
ftrain = "data/train.csv"
fval = "data/val.csv"
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
ftest = "data/test.csv"
foutput = "data/test_preds.csv"
model.predict(ftest, foutput)

# save the model
model.save('./tgb.model')

# load model and predict
model = tgb.load_model('./tgb.model')
model.predict(ftest, foutput)

```


## Reference

- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [SLIQ: A Fast Scalable Classifier for Data Mining](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.89.7734&rep=rep1&type=pdf)

- [GBDT算法原理与系统设计简介](http://wepon.me/files/gbdt.pdf)
