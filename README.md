# Introduction-to-TensorFlow-for-Artificial-Intelligence-Machine-Learning-and-Deep-Learning

API in TensorFlow called keras. Keras makes it really easy to define neural networks. A neural network is basically a set of functions which can learn patterns. Consider the following line of code.

#### model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) => Creates the deep learning model (architecture)

In keras, you use the word `Dense` to define a layer of connected neurons. There's only one dense here. So there's only one layer and there's only one unit in it, so it's a single neuron. Successive layers are defined in sequence, hence the word `Sequential`.

#### model.compile(optimizer='sgd', loss='mean_squared_error') => compiles the methods should be applied on the deep learning model

`model.compile()` takes in the optimization and the loss algorithms to be applied on the dataset to predict the output. It looks like a compilaiton of items to be applied on the model while executing.

#### model.fit(x, y, epochs=100) => trains the model with available inputs (x) and outputs (y)

`model.fit()` trains the deep learning model with the available input data (x) and answers/output data (y). `epochs` mentions how many times the training process should run. As the model predicts wrong values, depending upon the high loss values, the optimization algorithm optimizes the outcome. It is done by adding weights `w` to `x`. Finally, that derives one of the mostly used machine learning algorithm (linear regression) as`y = w.Tx`. And the weights are optimized by the optimization algorithm (most general one is gradient descent) which is `w = w - (alpha) * w` where `alpha` is the learning rate. In case of logistic regression the equation will be `y = w.Tx + b` where `b` is the bias. And then finally an activation function will be applied on it to predict a level of 0 or 1 by `sigmoid(y)`. So, in one sentence `model .fit() trains the neural network to fit one set of values to another`

#### model.predict(x-value) => predicts y-value (outputs)

Finally, the output `y-value` is predicted based on the `x-value`. Through the training process, the model learns the equation that lies between `y` and `x`. And to exactly predict the `y-value`, the model needs to be run for adequate times which is denoted by `epochs`. If the training period is not run adequate times, then the model will not be optimized appropriately and we might get erroneous result (less than the desired value in most cases). It can be denoted as `Underfit` problem.

Increase in epochs can lead to overfitting when loss can start increasing.

We can use callbacks() to control training. If the certain condition is met inside the function (constructor) of the callback class, then the training process will stop at that certain epoch.

{key : value} => this type of format with curly braces are used as dictionary in python. () => parentheses for tuples and [] => for lists.

model.summary() is a very handy tool to print out the model execution steps with output shapes!!!

In total, 2 rows and 2 columns always get eliminated from the original shape when the Conv2D (Convolution) layer is applied to the model. This happens due to the unavailability of the neighbouring pixels at the edges. So, for instance, if an image of shape (28, 28) is fed to the model and then convolution is applied, the output shape will be of (26, 26). *** See the handwritten notes for more detailed explanation ***

(2, 2) MaxPooling just halves(1/2) the total dimension of the image. So, if the image size is (26, 26, 1) then after applying MaxPooling the image size will be (13, 13, 1)

#### CNN (Convolutional Neural Network) is nothing but Deep Neural Network with Convolutions.

#### A little trick => whether it is the number of convolutional filters or dense layers, that should be in the form of multiples of 16.

















