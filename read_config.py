import configparser
from layers import * 
from net import NeuralNet
from train import train
from loss import *
from regularizers import *
from data_generation import generate_data,generate_1D_test
from optimize import Optimizer
from conv_layer import ConvolutionLayer,get_cnn_configurations
from matplotlib import pyplot as plt

"""

### CONFIG.INI EXAMPLE FILE

[TRAIN]
loss = CrossEntropy 
lrate = 0.01
reg = L1
alpha_reg = 0.01
num_epochs = 50
batch_size = 10
validation = True

[LAYERS]
l1 = LinearLayer size 10000,4
l3 = Softmax

[DATA]
n_samples = 200
size = 100
width = random
noise = 0.1
random_size = True
regular_polygons = False
flatten = True

"""

def get_net(config_layers):
    """
    Given the layers section in a config file, 
    return a network with the described layers

    !!Not yet support for Convolutional layers
    """

    layer_names = []
    linear_layer_sizes = []
    init_scale = 0.1
    c = 0

    conv_params = []

    for i,key in enumerate(config_layers):
        layer_list = config_layers[key].split(sep = ' ')

        if key == "init_scale":
            init_scale = float(config_layers[key])
        else:
            layer_names.append(layer_list[0])
            if "size" in layer_list:
                index = layer_list.index("size") + 1
                sizes = layer_list[index].split(sep = ',')

                sizes = [int(sizes[0]),int(sizes[1])]
                linear_layer_sizes.append(sizes)

            elif "ConvolutionLayer" in layer_list:
                index_size = layer_list.index("input_size") + 1
                if "scale" in layer_list:
                    index_scale = layer_list.index("scale") + 1
                else:
                    i_s = layer_list.index("s") + 1
                    i_p = layer_list.index("p") + 1
                    i_k = layer_list.index("k") + 1
                index_channels_o = layer_list.index("channels_out") + 1
                index_channels_i = layer_list.index("channels_in") + 1
                index_dim = layer_list.index("dim") + 1

                conv_params.append({"is":int(layer_list[index_size]),
                    "ci": int(layer_list[index_channels_i]),
                    "co": int(layer_list[index_channels_o]),
                    "dim": int(layer_list[index_dim]) })

                if "scale" in layer_list:
                    conv_params[-1]["sc"] = float(layer_list[index_scale])
                else:
                    conv_params[-1]["spk"] = {"s":int(layer_list[i_s]),
                                             "p":int(layer_list[i_p]),
                                             "k":int(layer_list[i_k])}


                


    layers = []

    for layer_name in layer_names:
        if layer_name == "LinearLayer":
            sizes = linear_layer_sizes.pop(0)
            layers.append(LinearLayer(sizes[0],sizes[1],init_scale))

        elif layer_name == "ConvolutionLayer":
            params = conv_params.pop(0)
            if "sc" in params:
                valid_configs = get_cnn_configurations(params["is"],params["sc"])[0]
                layers.append(ConvolutionLayer(params["ci"],params["co"],valid_configs["f"],params["is"],params["dim"],valid_configs["s"],valid_configs["p"],init_scale))
            else:
                layers.append(ConvolutionLayer(params["ci"],params["co"],params["spk"]["k"],params["is"],params["dim"],params["spk"]["s"],params["spk"]["p"],init_scale))
            
        else:
            layers.append(eval(layer_name)())
    net = NeuralNet(layers)
    return net


def get_training_vars(config_train):

    """
    Fetch training variables from configuration file.
    """

    loss = eval(config_train["loss"])

    if not eval(config_train["reg"]):
        reg = False
    else:
        alpha_reg = float(config_train["alpha_reg"])
        reg = eval(config_train["reg"])(alpha_reg)

    num_epochs = int(config_train["num_epochs"])
    batch_size = int(config_train["batch_size"])
    validation = bool(config_train["validation"])

    optimization_method = str(config_train["optimization_method"])
    optimization_params = [float(p) for p in config_train["optimization_params"].split(',')]
    
    

    return loss,reg,num_epochs,batch_size,validation,optimization_method,optimization_params

def get_data(config_data):
    """
    Get training and test data given parameters in config file. 
    Used for constructing the images of "randomized" shapes.
    """
    type = str(config_data["type"])
    n_samples_train = int(config_data["n_samples_train"])
    n_samples_test = int(config_data["n_samples_test"])
    size = int(config_data["size"])

    if type == "2D":


        if config_data["width"] == "random":
            width = "random"
        else:
            width = int(config_data["width"])
        if config_data["noise"] == "False":
            noise_strength = False
        else:
            noise_strength = float(config_data["noise"])
        random_size = bool(config_data["random_size"])
        regular_polygons = bool(config_data["regular_polygons"])
        flatten = bool(config_data["flatten"])

        data_train,labels_train = generate_data(n_samples = n_samples_train,
                                            size = size,
                                            width = width,
                                            noise_strength = noise_strength,
                                            random_size = random_size,
                                            regular_polygons = regular_polygons,
                                            flatten = True)

        data_test,labels_test = generate_data(n_samples = n_samples_test,
                                        size = size,
                                        width = width,
                                        noise_strength = noise_strength,
                                        random_size = random_size,
                                        regular_polygons = regular_polygons,
                                        flatten = True)

    elif type == "1D":
        data_train,labels_train = generate_1D_test(size,n_samples_train)
        data_test,labels_test = generate_1D_test(size,n_samples_train)


    return data_train,labels_train,data_test,labels_test,size,type

def compare_test(net,data_test,labels_test):
    """
    Run test data on net and print performance metrics.
    """


    cut_off = 0.4
    labels = {0 : "cross",1:"circle",2:"triangle",3:"square",4:"pentagon"}

    labels_pred = net.forward(data_test)

    error = np.abs(labels_pred - labels_test)
    error[np.where(error > cut_off)] = 1
    error[np.where(error < cut_off)] = 0



    error_per_class = np.sum(error,axis=1)

    wrong_index = np.where(np.sum(error,axis=0) > 0)
    wrong_x = data_test[wrong_index]

    for i,ce in enumerate(error_per_class):
        print(int(ce), " misclassifications for", labels[i])

    confusion_matrix = np.zeros((labels_test.shape[0],labels_test.shape[0]))
    for y_pred,y in zip(labels_pred.T,labels_test.T):
        index = np.where(y == 1)
        confusion_matrix[index] += y_pred

    confusion_matrix = np.round(confusion_matrix,2)

    print("\n","Confusion matrix")
    print(confusion_matrix, "\n")

    print("Index for misclassified images")
    print(wrong_index)

def plot_data(img_array):
    
    """
    Plotting array with training / test data (as n x n images)
    """
    n = int(np.sqrt(img_array.shape[0]))
    img = []
    for i in img_array.T:
        img.append(i.reshape(n,n))

    plt.figure(figsize = (4,4*len(img)))
    plt.imshow(np.concatenate(img))
    plt.show()

def plot_img_derivative(net,loss,img,label):
    n = int(np.sqrt(img.shape))
    pred = net.forward(img[np.newaxis].T)

    grad = loss.grad(pred,label[np.newaxis].T) 
    img_grad = net.backward(grad)
    
    img_grad_mask = np.ma.masked_where(np.abs(img_grad) > 0.2, img_grad)

    plt.imshow(img_grad.reshape(n,n),alpha=0.7,cmap="RdBu")
    



def read_config(filename,verbose = False):
    """
    Takes filename of config file and runs

    1) Training of network with data as requested (printing real time loss)
    2) Testing
    3) Print test metrics

    """

    config = configparser.ConfigParser()
    config.read(filename)

    net = get_net(config['LAYERS'])
    loss,reg,num_epochs,batch_size,validation,optimization_method,optimization_params = get_training_vars(config['TRAIN'])
    data_train,labels_train,data_test,labels_test,size,type = get_data(config['DATA'])


    print("Samples of the training data")
    if type == "2D":
        plot_data(data_test[:,0:labels_test.shape[0]])
    elif type == "1D":
        plt.imshow(data_train)
        plt.show()

    print("Epoch nr | Loss")
                                        
    train(net,
        inputs = data_train,
        targets = labels_train,
        num_epochs=num_epochs,
        batch_size=batch_size,
        loss=loss(),
        optimizer=Optimizer(optimization_method,optimization_params),
        regularizer=reg,
        validation= validation,
        verbose = verbose)
        
    if type == "2D":
        compare_test(net,data_test,labels_test)

        def plot_prediction(i):
            labels = {0 : "cross",1:"circle",2:"triangle",3:"square",4:"pentagon"}
            pred = net.forward(data_test)
            print(pred.shape)

            
            pred_label = np.where(pred[:,i] == np.max(pred[:,i]))[0][0]
            plot_title = labels[int(pred_label)] + " prob : " +  str(round(np.max(pred[:,i]),2))
            plt.title(plot_title)
            plt.imshow(data_test[:,i].reshape(size,size),cmap = "Greys")

            plot_img_derivative(net,loss(),data_test[:,i],labels_test[:,i])
            plt.colorbar()

            plt.show()

        
        
        return plot_prediction,net

    elif type == "1D":
        plt.figure()
        pred = net.forward(data_test)
        x = np.arange(0,len(labels_test[0]))
        plt.title("Error of 1D prediction")
        plt.plot(x,labels_test[0] - pred[0],'.')
        plt.show()

