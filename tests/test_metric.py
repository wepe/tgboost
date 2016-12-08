import numpy as np
from tgboost.metric import get_metric


preds = np.array([0.2,0.1,0.5,0.6,0.9,0.01,0.8,0.2,0.6,0.7])
labels = np.array([0,0,0,0,1,0,1,0,1,1])
print get_metric("error")(preds, labels)
print get_metric("acc")(preds, labels)


preds = np.zeros((10,))
labels = 2*np.ones((10,))
print get_metric("mse")(preds, labels)
print get_metric("mae")(preds, labels)