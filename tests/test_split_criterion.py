import numpy as np

from tests.split_criterion import xgb_score,mse_score,entropy_score,xgb_score_gain,mse_score_gain,entropy_score_gain
from tgboost.loss import SquareLoss,CustomizeLoss


def square_loss(pred, y):
    return 0.5*(y-pred)**2

print "--- score ---"
preds = np.array([15.0,9.0,17.0,15.0])
labels = np.array([10.0,10.0,20.0,20.0])
SquareLoss_object = SquareLoss()
print xgb_score(SquareLoss_object,preds,labels)
print mse_score(labels)
print entropy_score(np.array([0,1,0,0,1,0,0]))


print "--- score gain ---"

left_preds = np.array([15.0,9.0,17.0,15.0])
left_labels = np.array([10.0,10.0,20.0,20.0])
right_preds = np.array([10.0,20.0,30.0])
right_labels = np.array([20.0,20.0,20.0])

print xgb_score_gain(SquareLoss_object,left_preds,left_labels,right_preds,right_labels)
print xgb_score_gain(CustomizeLoss(square_loss),left_preds,left_labels,right_preds,right_labels)

print mse_score_gain(left_labels,right_labels)

left_labels = np.array([1,1,0,1])
right_labels = np.array([0,0])
print entropy_score_gain(left_labels,right_labels)

