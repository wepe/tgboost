import numpy as np


def accuracy(preds, labels):
    return np.mean(labels == preds.round())


def error(preds, labels):
    return 1.0 - accuracy(preds,labels)


metrics = {"acc": accuracy,
           "error": error}


def get_metric(eval_metric):
    return metrics[eval_metric]



