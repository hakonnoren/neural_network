from net import NeuralNet
from optimize import Optimizer
from loss import MSE,CrossEntropy
from optimize import Optimizer
import numpy as np
from matplotlib import pyplot as plt

debug = False


class Validation():
    def __init__(self,inputs,targets,validation_fraction):
        validation_size = int(inputs.shape[-1]//(1/validation_fraction))
        self.train_size = inputs.shape[-1] - validation_size
        self.v_inputs = inputs[...,self.train_size:]
        self.v_targets = targets[...,self.train_size:]
        self.v_errors = []
        self.counter = 0
        self.validation_freq = 2

    def get_validation_loss(self,net,loss):
        predicted = net.forward(self.v_inputs)
        self.v_errors.append(loss.loss(predicted,self.v_targets))

    def validate(self,net,loss):
        self.counter += 1
        if self.counter % self.validation_freq == 0:
            self.get_validation_loss(net,loss)

    

def train(net : NeuralNet,inputs,targets,num_epochs = 100,batch_size = 5,loss = CrossEntropy(), optimizer = Optimizer(),regularizer = False,validation = True,verbose = False):
    """
    inputs.shape = [sample_size,n_samples]
    targets.shape = [target_shape,n_samples]

    """


    reg_cost = lambda x: 0
    if  regularizer:
        reg_cost = regularizer.reg

    if validation:
        validator = Validation(inputs,targets,validation_fraction = 0.2)
        inputs = inputs[...,0:validator.train_size]
        targets = targets[...,0:validator.train_size]

    epoch_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        starts = np.arange(0,len(inputs[-1]),batch_size)
        np.random.shuffle(starts)

        for start in starts:
            end = start + batch_size
            num_batches = len(starts)
            predicted = net.forward(inputs[...,start:end],verbose = verbose)

            epoch_loss += loss.loss(predicted,targets[...,start:end])/num_batches #+ reg_cost(net)

            grad = loss.grad(predicted,targets[...,start:end])  
            net.backward(grad,verbose = verbose)

            if not isinstance(regularizer,bool):
                net.backward_regularizer(regularizer.grad_func)

            optimizer.step(net)
        if validation:
            validator.validate(net,loss)
        print(epoch,epoch_loss)
        epoch_losses.append(epoch_loss)

    if validation:
        plt.plot(np.linspace(0,num_epochs,num_epochs//validator.validation_freq),validator.v_errors,label="Validation")
    plt.plot(np.linspace(0,num_epochs,num_epochs),epoch_losses,label ="Training")
    plt.legend()


def run_validation(net,inputs,targets,loss):
    predicted = net.forward(inputs)
    loss = loss.loss(predicted,targets)
    return loss

