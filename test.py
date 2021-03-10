
from matplotlib import pyplot as plt
import numpy as np

class Diagnostic:
    def __init__(self,net,test_data,test_labels,labels_list,loss):
        self.net = net
        self.test_data = test_data
        self.test_labels = test_labels
        self.labels_list = labels_list
        self.pred = net.forward(test_data.T).T
        self.n_tests = test_labels.shape[0]
        self.n_classes = test_labels.shape[1]
        self.size = int(np.sqrt(test_data.shape[-1]))
        self.loss = loss
    
    def confusion_matrix(self,discrete = False):
        self.confusion_matrix = np.zeros((self.n_classes,self.n_classes))

        if discrete == True:
            pred_maxs = np.array([np.where(self.pred[i] == np.max(self.pred[i]))[0][0] for i in range(self.n_tests)])
            for pred_max,y in zip(pred_maxs,self.test_labels):
                one_hot = np.zeros((self.test_labels.shape[1]))
                one_hot[pred_max] = 1

                index = np.where(y == 1)
                self.confusion_matrix[index] += one_hot/(self.n_tests//self.n_classes)

        else:
            for y_pred,y in zip(self.pred,self.test_labels):
                index = np.where(y == 1)
                self.confusion_matrix[index] += y_pred/(self.n_tests//self.n_classes)

            
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        ax.set_xticks(np.linspace(0,self.n_classes,self.n_classes))
        ax.set_xticklabels(self.labels_list)
        ax.set_yticks(np.linspace(0,self.n_classes,self.n_classes))
        ax.set_yticklabels(self.labels_list)

        img = ax.imshow(self.confusion_matrix)
        fig.colorbar(img)


    def plot_img_derivative(self,img,label):
        pred = self.net.forward(img[np.newaxis].T)

        grad = self.loss.grad(pred,label[np.newaxis].T) 
        img_grad = self.net.backward(grad)
        img_grad_mask = np.ma.masked_where(np.abs(img_grad) > 0.2, img_grad)
        plt.imshow(img_grad.reshape(self.size,self.size),alpha=0.7,cmap="RdBu")

    def plot_prediction(self,i):
        pred_label = np.where(self.pred[i] == np.max(self.pred[i]))[0][0]
        correct = np.where(self.test_labels[i] == 1)[0][0]
        plot_title = self.labels_list[int(pred_label)] + " prob : " +  str(round(np.max(self.pred[i]),2)) + ", true : " + str(self.labels_list[correct])        
        plt.title(plot_title)
        plt.imshow(self.test_data[i].reshape(self.size,self.size),cmap = "Greys")
        self.plot_img_derivative(self.test_data[i],self.test_labels[i])
        plt.colorbar()

        plt.show()



    


"""



def compare_test(net,data_test,labels_test):


    cut_off = 0.4
    labels = {0 : "cross",1:"circle",2:"triangle",3:"square",4:"pentagon"}

    labels_pred = net.forward(data_test.T).T

    error = np.abs(labels_pred - labels_test)
    error[np.where(error > cut_off)] = 1
    error[np.where(error < cut_off)] = 0
    error_per_class = np.sum(error,axis=0)
    wrong_index = np.where(np.sum(error,axis=1) > 0)
    wrong_x = data_test[wrong_index]

    for i,ce in enumerate(error_per_class):
        print(int(ce), " misclassifications for", labels[i])

    confusion_matrix = np.zeros((labels_test.shape[1],labels_test.shape[1]))
    for y_pred,y in zip(labels_pred,labels_test):
        index = np.where(y == 1)
        confusion_matrix[index] += y_pred

    confusion_matrix = np.round(confusion_matrix,2)

    print("\n","Confusion matrix")
    print(confusion_matrix, "\n")

    print("Index for misclassified images")
    print(wrong_index)


    def plot_prediction(i):
        labels = {0 : "cross",1:"circle",2:"triangle",3:"square",4:"pentagon"}
        pred = net.forward(data_test.T).T
        
        pred_label = np.where(pred[i] == np.max(pred[i]))[0][0]
        plot_title = labels[int(pred_label)] + " prob : " +  str(round(np.max(pred[i]),2))
        plt.title(plot_title)
        plt.imshow(data_test[i].reshape(size,size),cmap = "Greys")

        plot_img_derivative(net,loss(),data_test[i],labels_test[i])
        plt.colorbar()

        plt.show()

def plot_data(img_array):
    


    n = int(np.sqrt(img_array.shape[1]))
    img = []
    for i in img_array:
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
    

"""