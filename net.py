from layers import Layer,LinearLayer
from conv_layer import ConvolutionLayer
from typing import Sequence
import numpy as np

class NeuralNet:

    """
    Defines Neural Net class.
    Initialized with sequence of layers.

    Calls the forward and backward passes of the layers sequentially
    """

    def __init__(self,layers: Sequence[Layer]):
        self.layers = layers


    def forward(self,inputs,verbose = False):
        """
        Forward pass calling on forward pass of layers in net.
        Exceptions for large inputs or nans in input
        If the previous layer is a convolution network, reshape output

        Returns output from final layer
        """

        prev_layer = 0
        for i,layer in enumerate(self.layers):
            
            if np.any(np.isnan(inputs)):
                print(inputs)
                raise Exception("Nan encountered layer no ", i, "of type ", layer.__class__.__name__)

            elif np.max(inputs) > 1e5:
                print(inputs)
                print("Large input in layer no", len(self.layers) - 1 - i, "of type ", layer.__class__.__name__)
                raise Exception("Large gradient in layer no", len(self.layers) - 1 - i, "of type ", layer.__class__.__name__)

            if isinstance(prev_layer,ConvolutionLayer):
                inputs = inputs.reshape(-1,inputs.shape[-1])


            prev_layer = layer


            if verbose:
                print("Input in :", layer.__class__.__name__, " = ", inputs.shape)

            inputs = layer.forward(inputs)

        return inputs

    def backward(self,grad,verbose = False):
        """
        Performs the backward pass calling backward passes of the layers.
        Input is gradient normally from loss function.

        If next layer is convolutional layer, transform gradient.
        """
        prev_layer = 0

        for i,layer in enumerate(reversed(self.layers)):

            if verbose:
                print("Grad in :", layer.__class__.__name__, " = ", grad.shape)
        
            if np.any(np.isnan(grad)):
                raise Exception("Nan encountered layer no ", len(self.layers) - 1 - i, "of type ", layer.__class__.__name__)
            
            elif np.max(grad) > 1000:
                print(grad)
                print("Large gradient in layer no", len(self.layers) - 1 - i, "of type ", layer.__class__.__name__)
                raise Exception("Large gradient in layer no", len(self.layers) - 1 - i, "of type ", layer.__class__.__name__)
            

            if isinstance(layer,ConvolutionLayer):
                grad = grad.reshape(layer.o_channels,layer.o_size**layer.d,layer.batch_size)

            elif isinstance(prev_layer,ConvolutionLayer):
                grad = grad.reshape(-1,grad.shape[-1])


            prev_layer = layer

            grad = layer.backward(grad)

        return grad

    def backward_regularizer(self,grad_func):
        """
        Backward pass wrt regularizer
        """

        for layer in self.layers:
            if isinstance(layer,LinearLayer):
                layer.backward_regularizer(grad_func)

    
