# General Neural Network class built with Numpy

Includes several common activation layers, standard feed forward layers and convolution layers with backpropagation.

To see examples of convolutional neural networks aiming at classifying basic geometric shapes, run *tests.ipynb* or run the configuration files directly in your favourite python terminal. This is achived by running *plot_pred,net = read_config("config_cnn_2d.INI")*. The returned *plot_pred* function could be run to visualize how different pixles affects the class prediction. 

This is done by plotting the derivative of the loss function with respect to the input, aiming at achieving some basic explainability. See examples below.

Explainability plot of geometric shapes using derivative of loss function

![alt text](https://raw.githubusercontent.com/hakonnoren/neural_network/main/examples/explain_geom_1.png)

![alt text](https://raw.githubusercontent.com/hakonnoren/neural_network/main/examples/explain_geom_2.png)

Explainability plot of letters using derivative of loss function (this example is not included in this repo)

![alt text](https://raw.githubusercontent.com/hakonnoren/neural_network/main/examples/explain_letter_1.png)

![alt text](https://raw.githubusercontent.com/hakonnoren/neural_network/main/examples/explain_letter_2.png)

This library includes a generator of basic geometric shapes, or crosses and polygons with added noise.

![alt text](https://raw.githubusercontent.com/hakonnoren/neural_network/main/examples/geom.png)

