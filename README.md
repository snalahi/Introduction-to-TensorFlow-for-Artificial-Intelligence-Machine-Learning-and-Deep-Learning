# Introduction-to-TensorFlow-for-Artificial-Intelligence-Machine-Learning-and-Deep-Learning

API in TensorFlow called keras. Keras makes it really easy to define neural networks. A neural network is basically a set of functions which can learn patterns. Consider the following line of code.

#### model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

In keras, you use the word `Dense` to define a layer of connected neurons. There's only one dense here. So there's only one layer and there's only one unit in it, so it's a single neuron. Successive layers are defined in sequence, hence the word `Sequential`.

#### model.compile(optimizer='sgd', loss='mean_squared_error')

`model.compile()` takes in the optimization and the loss algorithms to be applied on the dataset to predict the output. It looks like a compilaiton of items to be applied on the model while executing.

#### model.fit(x, y, epochs=100)

`model.fit()` trains the deep learning model with the available input data (x) and answers/output data (y). `epochs` mentions how many times the training process should run. As the model predicts wrong values, depending upon the high loss values, the optimization algorithm optimizes the outcome. It is done by adding weights `w` to `x` and also a bias value `b`. Finally, that derives the mostly used machine learning algorithm (logistic regression) as
`y = w.Tx + b`. And the weights are optimized by the optimization algorithm (most general one is gradient descent) which is `w = w - (alpha) * w` where `alpha` is the learning rate.

#### model.predict(x-value)

Finally, the output `y-value` is predicted based on the `x-value`. Through the training process, the model learns the equation that lies between `y` and `x`. And to exactly predict the `y-value`, the model needs to be run for adequate times which is denoted by `epochs`. If the training period is not run adequate times, then the model will not be optimized appropriately and we might get erroneous result (less than the desired value in most cases). It can be denoted as `Underfit` problem.



