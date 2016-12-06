from tgboost.loss import CustomizeLoss, SquareLoss, LogisticLoss
import numpy as np
import autograd.numpy as anp


print "---- test SquareLoss ----"
preds = np.array([15.0,9.0,17.0,15.0])
labels = np.array([10.0,10.0,20.0,20.0])

SquareLoss_object = SquareLoss()
print SquareLoss_object.grad(preds, labels)
print SquareLoss_object.hess(preds, labels)

print "---- test CustomizeLoss, square_loss ----"
def square_loss(pred, y):
    return 0.5*(y-pred)**2

preds = np.array([15.0,9.0,17.0,15.0])
labels = np.array([10.0,10.0,20.0,20.0])
Customize_SquareLoss_object = CustomizeLoss(square_loss)
print Customize_SquareLoss_object.grad(preds, labels)
print Customize_SquareLoss_object.hess(preds, labels)



print "---- test LogisticLoss ----"
preds = np.array([0.1,0.9,0.5,0.8])
labels = np.array([0,0,1,1])
LogisticLoss_object = LogisticLoss(l2_regularization=0.2)
print LogisticLoss_object.grad(preds, labels)
print LogisticLoss_object.hess(preds, labels)


print "---- test CustomizeLoss, logistic_loss ----"
def logistic_loss(pred,y):
    return -(y*anp.log(pred) + (1-y)*anp.log(1-pred))

preds = np.array([0.1,0.9,0.5,0.8])
labels = np.array([0,0,1,1])
Customize_LogisticLoss_object = CustomizeLoss(logistic_loss,0.2)
print Customize_LogisticLoss_object.grad(preds, labels)
print Customize_LogisticLoss_object.hess(preds, labels)

