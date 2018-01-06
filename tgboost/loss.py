from autograd import elementwise_grad
import numpy as np

elementwise_hess = lambda func: elementwise_grad(elementwise_grad(func))


class BaseLoss(object):
    def __init__(self):
        pass

    def grad(self, preds, labels):
        raise NotImplementedError()

    def hess(self, preds, labels):
        raise NotImplementedError()


class CustomizeLoss(BaseLoss):
    """
    define your loss function:
         square_loss = lambda pred,y:0.5*(y-pred)**2
    your_loss_object = CustomizeLoss(square_loss)

    TODO: when customize loss is for regression, change the transform function

    """
    def __init__(self, loss):
        super(CustomizeLoss, self).__init__()
        self.loss = loss

    def grad(self, preds, labels):
        preds = self.transform(preds)
        return elementwise_grad(self.loss)(preds, labels)

    def hess(self, preds, labels):
        preds = self.transform(preds)
        return elementwise_hess(self.loss)(preds, labels)

    def transform(self, preds):
        """
        logistic tranformation
        """
        return 1.0/(1.0+np.exp(-preds))


class SquareLoss(BaseLoss):
    """
    SquareLoss_l2regularization = SquareLoss(10)
    """
    def transform(self, preds):
        return preds

    def grad(self, preds, labels):
        return preds-labels

    def hess(self, preds, labels):
        return np.ones_like(labels)


class LogisticLoss(BaseLoss):
    """
    label is {0,1}
    grad = (1-y)/(1-pred) - y/pred
    hess = y/pred**2 + (1-y)/(1-pred)**2
    """
    def transform(self, preds):
        """
        logistic tranformation
        """
        return np.clip(1.0/(1.0+np.exp(-preds)),0.00001,0.99999)

    def grad(self, preds, labels):
        preds = self.transform(preds)
        return (1-labels)/(1-preds) - labels/preds

    def hess(self, preds, labels):
        preds = self.transform(preds)
        return labels/np.square(preds) + (1-labels)/np.square(1-preds)
