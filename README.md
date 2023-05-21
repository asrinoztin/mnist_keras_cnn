# Summary
In this study, the data set known as mnist was used. The aim of the study is to teach and predict the handwritten numbers in each of the images in the data set, which can take one of 10 unique values as 0 to 9 inclusive in total, by working in pixel size, to the model.

# Design Choices
Before image classification, we should know the dataset well and make it available to the model. This optimization process is called feature learning. First, the image is made easier to process using the convolution layer. After this process, in the pooling layer, as in the convolution layer, the size reduction process is applied by moving the kernel over the pixels, which we know from image processing, in order to reduce the processing power. After these processes are repeated enough and the data is made easy to process, the outputs of the feature learning process are given to the model as input.

![146385526-d7cb9b40-4631-4bb1-83ad-0842b07faf24](https://user-images.githubusercontent.com/58219688/146405076-4520c496-6293-44f6-88ef-15fadfcd30a2.png)

# Layer Properties
## tf.keras.layers.Conv2D
Available at https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

Keras info: This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

Role: The role of the Convolutional Network is to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction.

## tf.keras.layers.BatchNormalization
Available at https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization 

Keras info: Layer that normalizes its inputs.

Role: BatchNormalization has the effect of stabilizing the learning process and also reducing the number of epochs required to train the artificial neural networks models.

## tf.keras.layers.Flatten
Available at https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten 

Keras info: Flattens the input. Does not affect the batch size.

Role: It basically vectorizes (flattens) the input and returns it as a one dimensional matrix (vector).

## tf.keras.layers.Dense
Available at https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense 

Keras info: Just your regular densely-connected NN layer.

Role: The role is to learn/generalize the non-linear function.

# References
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D 
https://www.tensorflow.org/tutorials/images/cnn 
https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalMaxPool2D
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout 
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
https://www.tensorflow.org/tutorials/images/classification 
