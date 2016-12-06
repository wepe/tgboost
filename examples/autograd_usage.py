import autograd.numpy as anp
from autograd import grad,hessian,elementwise_grad
import numpy as np


#scale value function
def square_loss(pred,y):
    return 0.5*(y-pred)**2

def logistic_loss(pred,y):
    return -(y*anp.log(pred) + (1-y)*anp.log(1-pred))

# square_loss_grad(pred,y)
# square_loss_hess(pred,y)
square_loss_grad = grad(square_loss)  # square_loss with respect to  pred, grad = pred-y
square_loss_hess = hessian(square_loss)  # square_loss_grad with respect to pred, hess = 1

print square_loss_grad(0.0, 0.5)  # -0.5
print square_loss_hess(0.0, 0.5)  # 1

# logistic_loss_grad(pred,y)
# logistic_loss_hess(pred,y)
logistic_loss_grad = grad(logistic_loss)  # logistic_loss with respect to  pred, grad = (1-y)/(1-pred) - y/pred
logistic_loss_hess = hessian(logistic_loss)  #logistic_loss_grad with respect to  pred, hess = y/pred**2 + (1-y)/(1-pred)**2


print logistic_loss_grad(0.2, 0)  # 1.25
print logistic_loss_hess(0.2, 0)  # 1.25
print logistic_loss_grad(0.6, 1)  # -1.66
print logistic_loss_hess(0.6, 1)  # 2.777

# elementwise_grad, elementwise_hess
elementwise_hess = lambda func:elementwise_grad(elementwise_grad(func))

preds = np.array([0.5,0.2,0.4,0.5])
y = np.array([1,0,0,1])
print elementwise_grad(logistic_loss)(preds, y)
print elementwise_hess(logistic_loss)(preds, y).sum()



