import numpy as np


def accuracy(preds, labels):
    return np.mean(labels == preds.round())


def error(preds, labels):
    return 1.0 - accuracy(preds,labels)


def mean_square_error(preds, labels):
    return np.mean(np.square(preds - labels))


def mean_absolute_error(preds, labels):
    return np.mean(np.abs(preds - labels))

metrics = {"acc": accuracy,
           "error": error,
           "mse": mean_square_error,
           "mae": mean_absolute_error}


def get_metric(eval_metric):
    return metrics[eval_metric]



