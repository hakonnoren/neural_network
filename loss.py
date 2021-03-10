import numpy as np

class MSE:
    """
    MSE loss function with gradient
    """
    def loss(self,predicted,actual):
        return np.sum((predicted-actual)**2)

    def grad(self,predicted,actual):
        return 2*(predicted - actual)

class CrossEntropy:
    

    """
    Cross entropy loss function with gradient.
    Throws exceptions when zero in denominator
    """

    def loss(self,predicted,actual):
        loss = 0
        batch_size = predicted.shape[-1]
        
        #for p,a in zip(predicted.T,actual.T):

        loss = -1*np.sum(actual*np.log(predicted))
        return loss/batch_size
        
    def grad(self,predicted,actual):
        epsilon = 1e-8
        gradient = np.zeros(predicted.shape)
        batch_size = predicted.shape[-1]

        if np.any(predicted == 0):
            print(predicted)
            #raise Exception("Zero in denominator of Cross Entropy gradient")

        gradient[np.nonzero(actual)] = -1*actual[np.nonzero(actual)]/(predicted[np.nonzero(actual)])
        
        if np.any(np.isnan(gradient)):
            raise Exception("Nan in Cross Entropy gradient")

        return gradient

