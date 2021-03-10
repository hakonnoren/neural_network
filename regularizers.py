import numpy as np

class L2:

    """
    Takes parameter setting the regularization penalty
    L2 regularization given L2 norm, ord frobenious norm for matrices
    Defines regularization function and its gradient.
    """

    def __init__(self,alpha):
        self.alpha = alpha

    def reg(self,net):
        sum = 0
        for w in net.get_weights():
            sum += np.linalg.norm(w,ord = 'fro')**2
        return self.alpha*sum*0.5

    def grad_func(self,w):
        return self.alpha*w

class L1:

    """
    Takes parameter setting the regularization penalty
    L1 regularization given L1 norm.
    Defines regularization function and its gradient.
    """

    def __init__(self,alpha):
        self.alpha = alpha

    def reg(self,net):
        sum = 0
        for w in net.get_weights():
            sum += np.linalg.norm(w.flatten(),ord = 1)**2
        return self.alpha*sum*0.5

    def grad_func(self,w):
        return self.alpha*np.sign(w)