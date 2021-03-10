import numpy as np
from matplotlib import pyplot as plt
from layers import Layer
from itertools import product
import time

def get_cnn_configurations(n,rho,verbose = False):

    """
    Input:
        n : input size of data to convolution layer
        rho : factor to find the output_size ie. input_size * rho = output_size

    Find valid strides, paddings and kernelsizes for doing convolution.
    """

    ss = range(1,10)
    fs = range(2,10)

    configs = []

    if verbose:
        p = 0.5*(fs[0]-n-ss[0]+int(rho*n)*ss[0])
        t = (n-fs[0]+2*p)/ss[0] + 1
        print("Output size = ", int(t))

    for s in ss:
        for f in fs:

            p = 0.5*(f-n-s+int(rho*n)*s)
            t = (n-f+2*p)/s
            if int(t) - t == 0 and int(p) - p == 0 and p>=0 and p < 10:
                
                configs.append({"s":s,"p":int(p),"f":f})
                if verbose:
                    p
                    print("s = ", s, ", p = ", int(p), ", f = ",f)

    return configs


class ConvolutionLayer(Layer):

    def __init__(self,i_channels,o_channels,k_size,i_size,dim,stride,padding,init_scale = 0.1):
        """
        initialized with:
            i_channels : number of input channels
            o_channels : number of output channels (layers in the kernel)
            k_size : size of kernel
            i_size : size of one dimension of input data
            dim : dimension of input data
            stride : step length when convolving
            padding : padding to ensure correct sizes
            init_scale : multiplication factor when initializing kernel weights

        """

        self.d = dim
        self.stride = stride
        self.padding = padding
        self.i_size = i_size
        self.o_channels = o_channels
        self.i_channels = i_channels
        self.k_size = k_size
        #Finds outputsize and throws error if output size is rational 
        o_size = (i_size-k_size + 2*padding)/stride + 1
        assert o_size - int(o_size) == 0, "Error: Rational output sizes!"
        self.o_size = int(o_size)
        self.ks = []
        self.k_grads = []

        #initializing flat kernels (convolution applied as dot product)
        self.flat_kernels = np.random.random((o_channels,i_channels*k_size**self.d))*init_scale #- 0.5*init_scale
        self.s_k = np.zeros((o_channels,i_channels*k_size**self.d))
        self.r_k = np.zeros((o_channels,i_channels*k_size**self.d))
        self.t = 0

        self.itk = 0
        self.kti = 0

        #mapping indicies to transform data when doing convolutions as dot products
        self.index_map_to_kernel = self.make_index_mapping_to_kernelsize()
        self.index_map_to_input = self.make_index_mapping_to_inputsize()


    #storing and plotting parameters as for linear layers
    def store_params(self,norm_ord):
        self.ks.append(np.linalg.norm(self.flat_kernels,ord=norm_ord))
        self.k_grads.append(np.linalg.norm(self.flat_kernels_grad,ord=norm_ord))

    #storing and plotting parameters as for linear layers
    def plot_params(self):
        xs = lambda y: np.linspace(0,len(y),num = len(y))
        def plot(data,title):
            plt.plot(xs(data),data)
            plt.title(title)
            plt.show()

        plot(self.ks,"ks")
        plot(self.k_grads,"k_grads")

    #plotting kernels
    def plot_kernel(self):
        kernels = self.flat_kernels.reshape(self.k_size*self.o_channels,self.k_size*self.i_channels)
        plt.imshow(kernels)
        plt.show()

    #Making index mapping to transform input data to kernel sized data enabling dot product convolution
    def make_index_mapping_to_kernelsize(self):
        k = 0
        index_slice = slice(0,self.i_size+2*self.padding)
        index = np.mgrid[tuple(index_slice for d in range(self.d))]

        flat_indicies = np.zeros((self.d,self.k_size**self.d,self.o_size**self.d),dtype=int)
        for i in product(range(self.o_size),repeat=self.d):

            flat_indicies[:,:,k] = index[tuple( [slice(None)] + [slice(i[d]*self.stride,i[d]*self.stride + self.k_size) for d in range(self.d)] )].reshape(self.d,self.k_size**self.d)
            k+=1

        return flat_indicies

    #Reverse mapping to the above
    def make_index_mapping_to_inputsize(self):
        k = 0
        dim = self.i_size+2*self.padding

        index = np.mgrid[0:self.k_size**self.d,0:self.o_size**self.d]
        input_indicies = np.zeros(( tuple([2] + [dim for i in range(self.d)])),dtype=int)

        for i in product(range(self.o_size),repeat=self.d):
            input_indicies[ tuple([slice(None)] +[slice(i[d]*self.stride,i[d]*self.stride + self.k_size) for d in range(self.d)])] = index[:,:,k].reshape(tuple([2] + [self.k_size for d in range(self.d)]))

            k+=1
        return input_indicies

    #Transforming data from input shape to kernelsized data for dot product convolution
    def input_to_kernelsize_fast(self,input):

        flat_input = input[tuple([slice(None)] + [tuple(self.index_map_to_kernel[d]) for d in range(self.d)] + [slice(None)])].reshape(self.i_channels*self.k_size**self.d,
        self.o_size**self.d,self.batch_size)

        return flat_input

    #Reverse transformation as above 
    def kernelsize_to_input_fast(self,kernelsize):

        index = self.index_map_to_input
        input = kernelsize.reshape(self.i_channels,self.k_size**self.d,self.o_size**self.d,self.batch_size)[tuple([slice(None)] + [tuple(self.index_map_to_input[d]) for d in range(self.d)] + [slice(None)])]

        if self.padding > 0:
            input = input[tuple([slice(None)] + [slice(self.padding,-self.padding) for d in range(self.d)] + [slice(None)])]

        if self.d == 1:
            output = input[:,:,0,:].reshape(self.i_channels,self.i_size,self.batch_size)
            input = output

            return input

        else:
            return input.reshape(self.i_channels,self.i_size**self.d,self.batch_size)
    
    #Applying neccessary padding to data before doing convolution
    def apply_padding(self,input):
        shape = np.array(input.shape)
        shape[1:self.d + 1] += 2*self.padding
        padded = np.zeros(shape)
        padded[tuple([slice(None)] + [slice(self.padding,-self.padding) for d in range(self.d)] + [slice(None)])] = input
        input = padded
        return input

    def forward(self,input):
        """
        Assume input on format
        Nd_array.shape = (channels,input_size, .. ,input_size,batch_size)

        Returns output on format
        Nd_array.shape = (o_channels,o_size**self.d,batch_size)
        """
        self.batch_size = input.shape[-1]
        input = input.reshape(tuple( [self.i_channels] + [self.i_size for d in range(self.d)] + [self.batch_size]))

        if self.padding > 0:
            input = self.apply_padding(input)

        self.flat_input = self.input_to_kernelsize_fast(input)
        
        output = np.einsum('ci,iob->cob',self.flat_kernels,self.flat_input)
        return output

    def backward(self,grad):

        """
        c = o_channels
        o = output_size**self.d
        i = i_channels*k_size**self.d
        b = batch_size
        """
        self.flat_kernels_grad = np.einsum('cob,iob->ci',grad,self.flat_input)
        self.store_params(norm_ord = 2)


        return self.kernelsize_to_input_fast(np.einsum('cob,ci->iob',grad,self.flat_kernels))

    def step(self,learning_rate):
        self.flat_kernels -= learning_rate*self.flat_kernels_grad

    def step_adam(self,rho_1,rho_2,epsilon):
        self.t += 1
        self.delta = 1e-8
        """
        Performs the gradient descent step given learning rate
        """
        self.s_k = rho_1*self.s_k + (1-rho_1)*self.flat_kernels_grad
        self.r_k = rho_2*self.r_k + (1-rho_2)*self.flat_kernels_grad*self.flat_kernels_grad

        self.flat_kernels_grad -= epsilon*(self.s_k/(1-rho_1**self.t))/(np.sqrt(self.r_k/(1-rho_2**self.t)) + self.delta)


