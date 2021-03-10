import numpy as np
from matplotlib import pyplot as plt



class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError

class LinearLayer(Layer):

    """
    Linear Layer for densely connected neural networks.
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights and biases
        """

        self.w = np.random.randn(output_size,input_size)*init_scale #- init_scale
        self.b = np.random.randn(output_size,1)*init_scale #- init_scale

        self.w_grad = np.zeros((output_size,input_size))
        self.b_grad = np.zeros((output_size,1))

        self.s_w = np.zeros((output_size,input_size))
        self.s_b = np.zeros((output_size,1))
        self.r_w = np.zeros((output_size,input_size))
        self.r_b = np.zeros((output_size,1))
        self.t = 0

        self.ws = []
        self.bs = []
        self.w_grads = []
        self.b_grads = []

    def forward(self,inputs):
        """
        Computes the affine transformation of the forward pass
        Stores input and returns output
        """

        self.inputs = inputs
        return np.dot(self.w,inputs) + self.b

    def store_params(self,norm_ord):
        """
        Function storing the norm of the parameters b and w for plotting
        """

        self.ws.append(np.linalg.norm(self.w,ord=norm_ord))
        self.bs.append(np.linalg.norm(self.b,ord=norm_ord))
        self.w_grads.append(np.linalg.norm(self.w_grad,ord=norm_ord))
        self.b_grads.append(np.linalg.norm(self.b_grad,ord=norm_ord))

    def backward(self,grad):
        """
        Performs backward pass.
        Input is gradient for the downstream layer.
        Calculates loss gradient wrt weights w, biases b and input to layer.
        """
        
        #in einsum 'o' is output_size, 'i' is input_size, 'b' is batch_size

        self.w_grad = np.einsum('ob,ib->oi',grad,self.inputs)
        self.b_grad = np.sum(grad,axis=1)[np.newaxis].T
        self.store_params(norm_ord = 2)

        return np.einsum('oi,ob->ib',self.w,grad)

    def backward_regularizer(self,grad_func):
        """
        Contribution to gradient of w regularizer
        """
        self.w_grad += grad_func(self.w)

    def step(self,learning_rate):
        """
        Performs the gradient descent step given learning rate
        """
        self.w -= learning_rate*self.w_grad
        self.b -= learning_rate*self.b_grad

    def step_adam(self,rho_1,rho_2,epsilon):
        self.t += 1
        self.delta = 1e-8
        """
        Performs the gradient descent step given learning rate
        """
        self.s_w = rho_1*self.s_w + (1-rho_1)*self.w_grad
        self.s_b = rho_1*self.s_b + (1-rho_1)*self.b_grad
        self.r_w = rho_2*self.r_w + (1-rho_2)*self.w_grad*self.w_grad
        self.r_b = rho_2*self.r_b + (1-rho_2)*self.b_grad*self.b_grad

        self.w -= epsilon*(self.s_w/(1-rho_1**self.t))/(np.sqrt(self.r_w/(1-rho_2**self.t)) + self.delta)
        self.b -= epsilon*(self.s_b/(1-rho_1**self.t))/(np.sqrt(self.r_b/(1-rho_2**self.t)) + self.delta)

    def get_params_grads(self):
        yield self.w,self.b,self.w_grad,self.w

    def plot_params(self):
        """
        Plots norm of w and b for every backward pass through learning
        """

        xs = lambda y: np.linspace(0,len(y),num = len(y))
        def plot(data,title):
            plt.plot(xs(data),data)
            plt.title(title)
            plt.show()

        plot(self.ws,"ws")
        plot(self.bs,"bs")
        plot(self.w_grads,"w_grads")
        plot(self.b_grads,"b_grads")

    def plot_weights(self):
        i = int(np.sqrt(self.w.shape[1]))
        o = self.w.shape[0]

        weights = self.w.reshape(i*o,i)
        
        plt.figure(figsize = (4,4*o))
        plt.imshow(weights)
        plt.show()

    

class ActivationLayer(Layer):
    """
    Base class for activation layers
    """
    def __init__(self,f,f_prime):
        self.f = f
        self.f_prime = f_prime

    def forward(self,inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self,grad):
        return np.einsum('iob,ib->ob',self.f_prime(self.inputs), grad)


#Definition of softmax with exeption when denominator is zero
def softmax(x):
    
    sum_exp = np.sum(np.exp(x),axis=0)[np.newaxis]

    if np.any(sum_exp == 0):
        print(sum_exp)
        raise Exception("Zero in denominator in Softmax")

    elif np.any(np.isnan(np.exp(x)/sum_exp)):
        print(np.exp(x)/sum_exp)
        raise Exception("Nans out from Softmax")

    return np.exp(x)/sum_exp

#Definition of softmax_prime with batchwise loop
def softmax_prime(x):
    S = softmax(x)
    batch_size = x.shape[-1]

    grad = np.zeros((x.shape[0],x.shape[0],batch_size))
    for i in range(batch_size):
        grad[:,:,i] = np.diag(S[:,i]) - np.dot(S[:,i][np.newaxis].T,S[:,i][np.newaxis])
    return grad

class Softmax(ActivationLayer):
    def __init__(self):
        super().__init__(softmax,softmax_prime)


#Definition of Relu activation layer with forward and backward passes explicitly
class Relu(ActivationLayer):
    def __init__(self):
        return

    def relu(self,x):
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,inputs):
        self.inputs = inputs
        return self.relu(inputs)

    def backward(self,grad):
        return grad * np.where(self.inputs > 0, np.ones(self.inputs.shape), np.zeros(self.inputs.shape))

#Definition of Tanh activation layer with forward and backward passes explicitly
class Tanh(ActivationLayer):
    def __init__(self):
        return

    def forward(self,inputs):
        self.inputs = inputs
        return np.tanh(inputs)

    def backward(self,grad):
        return grad * (1 - np.tanh(self.inputs)**2)

#Definition of Sigmmoid activation layer with forward and backward passes explicitly
class Sigmoid(ActivationLayer):
    def __init__(self):
        return
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def forward(self,inputs):
        self.inputs = inputs
        return self.sigmoid(inputs)

    def backward(self,grad):
        return grad * (self.sigmoid(self.inputs)*(1-self.inputs))

