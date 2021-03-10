Run tests_p2.ipynb 

or run the following in your favourite Python terminal

#################

from read_config import read_config

#This might be re run multiple times as the network not always manages to learn
read_config("config_cnn_1d.INI")

plot_pred,net = read_config("config_cnn_2d.INI")
plot_pred(3)
net.layers[0].plot_kernel()

plot_pred,net = read_config("config_cnn_2d_1d.INI")
