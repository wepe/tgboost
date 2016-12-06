from collections import defaultdict
import pandas as pd
import numpy as np
from loss import SquareLoss, LogisticLoss, CustomizeLoss
from tree import Tree
from metric import get_metric


class TGBoost(object):
    """
    Tiny Gradient Boosting
    """
    def __init__(self):
        self.trees = []
        self.eta = None
        self.num_boost_round = None
        self.first_round_pred = None
        self.loss = None
        self.max_depth = None
        self.rowsample = None
        self.colsample_bytree = None
        self.colsample_bylevel = None
        self.l2_regularization = None
        self.min_sample_split = None
        self.gamma = None
        self.num_thread = None
        self.feature_importance = defaultdict(lambda: 0)

    def fit(self,X,y,eta=0.01,num_boost_round=1000,max_depth=5,rowsample=0.8,colsample_bytree=0.8,colsample_bylevel=0.8,
            min_sample_split=10,loss="logisticloss",l2_regularization=1.0,gamma=0.1,num_thread=-1,eval_metric=None):

        """
        :param X: pandas.core.frame.DataFrame
        :param y: pandas.core.series.Series
        :param eta: learning rate
        :param num_boost_round: number of boosting round
        :param max_depth: max depth of each tree
        :param rowsample: row sample rate when building a tree
        :param colsample_bytree: column sample rate when building a tree
        :param colsample_bylevel: column sample rate when spliting each tree node,
                                  the number of features = total_features*colsample_bytree*colsample_bylevel
        :param min_sample_split: min number of samples in a leaf node
        :param loss: loss object
                     logisticloss,squareloss, or customize loss
        :param l2_regularization: lambda
        :param gamma: gamma
        :param seed: random seed
        :param num_thread: number of thread to parallel
        :param eval_metric: evaluation metric, provided: "accuracy"
        """
        self.eta = eta
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.rowsample = rowsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.l2_regularization = l2_regularization
        self.gamma = gamma
        self.min_sample_split = min_sample_split
        self.num_thread = num_thread
        self.eval_metric = eval_metric

        if loss == "logisticloss":
            self.loss = LogisticLoss(l2_regularization)
        elif loss == "squareloss":
            self.loss = SquareLoss(l2_regularization)
        else:
            try:
                self.loss = CustomizeLoss(loss, l2_regularization)
            except:
                raise NotImplementedError("loss should be 'logisticloss','squareloss', or customize loss function")

        self.first_round_pred = y.mean()

        # Y stores label, y_pred, grad, hess
        Y = pd.DataFrame(y.values, columns=['label'])  # only one column "label"
        Y['y_pred'] = self.first_round_pred
        Y['grad'] = self.loss.grad(Y.y_pred.values, Y.label.values)
        Y['hess'] = self.loss.hess(Y.y_pred.values, Y.label.values)

        for i in range(self.num_boost_round):
            # sample samples and features to train current tree
            data = X.sample(frac=self.colsample_bytree, axis=1)
            data = pd.concat([data,Y],axis=1)
            data = data.sample(frac=self.rowsample, axis=0)
            Y_selected = data[['label', 'y_pred', 'grad', 'hess']]
            X_selected = data.drop(['label', 'y_pred', 'grad', 'hess'], axis=1)

            # train current tree
            tree = Tree()
            tree.fit(X_selected, Y_selected, max_depth=self.max_depth,
                     colsample_bylevel=self.colsample_bylevel,min_sample_split=self.min_sample_split,
                     l2_regularization=self.l2_regularization, gamma=self.gamma, num_thread=self.num_thread)

            # predict the whole dataset and update y_pred,grad,hess
            preds = tree.predict(X)
            Y['y_pred'] += self.eta * preds
            Y['grad'] = self.loss.grad(Y.y_pred.values, Y.label.values)
            Y['hess'] = self.loss.hess(Y.y_pred.values, Y.label.values)

            if self.eval_metric is not None:
                try:
                    mertric_func = get_metric(self.eval_metric)
                except:
                    raise NotImplementedError("The given eval_metric is not provided")
                metric_value = mertric_func(self.loss.transform(Y.y_pred.values), Y.label.values)
                print "TGBoost round {iteration}, {eval_metric} is {metric_value}".format(iteration=i, eval_metric=self.eval_metric, metric_value=metric_value)
            else:
                print "TGBoost round {iteration}"

            # update feature importance
            for k in tree.feature_importance.iterkeys():
               self.feature_importance[k] += tree.feature_importance[k]

            self.trees.append(tree)

    def predict(self, X):
        preds = np.zeros((X.shape[0],))
        for tree in self.trees:
            preds += self.eta * tree.predict(X)

        preds += self.first_round_pred
        return self.loss.transform(preds)









