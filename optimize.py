from net import NeuralNet
from layers import LinearLayer
from conv_layer import ConvolutionLayer

class Optimizer():

    """
    Very basic gradient descent optimizer calling the gradient descent step
    of layers with parameters with a given learning rate.

    Method chooses between Adam and standard gradient descent.
    Optimization parameters are for Adam : [rho_1,rho_2,epsilon]
    for gradient descent : [learning_rate]
    """

    def __init__(self,method = "adam",optimization_params = [0.9,0.999,0.001]):
        self.method = method
        self.optimization_params = optimization_params

    def step(self,net):
        for layer in net.layers:
            if isinstance(layer,(LinearLayer,ConvolutionLayer)):
                if self.method == "adam":
                    layer.step_adam(
                    self.optimization_params[0],
                    self.optimization_params[1],
                    self.optimization_params[2])
                else:
                    layer.step(self.optimization_params[0])
