import numpy as np


def accuracy(preds, labels):
    return np.mean(labels == preds.round())


def auc(preds,labels):
    pass


metrics = {"acc": accuracy,
           "auc": auc}


def get_metric(eval_metric):
    return metrics[eval_metric]



