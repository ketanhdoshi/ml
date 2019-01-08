# --------------------------------------------------------------------------------------------
# Basic Neural Network from scratch. The basic building block is the layer which contains
# the bulk of the logic for forward and back propagation. It then constructs the network layer 
# by layer. The activation function, the cost function, and the initialisation of the weights
# and biases are passed in to network. The network can work with any function and does not
# make any assumption about what those functions are.
# Currently we have implemented:
# Weight and Bias Initialisation: Normal Guassian distribution
# Activation Function: Sigmoid. Could implement ReLU and Softmax as well
# Cost function: Mean Square Error and Cross Entropy
# --------------------------------------------------------------------------------------------

import numpy as np
from mnist_loader import load_data_wrapper

# --------------------------------------------------------------------------------------------
# Dummy Weight and Bias Initialisation function used for testing
# --------------------------------------------------------------------------------------------
class TestWeightBiasInit(object):

    @staticmethod
    def fn(size_in, size):
        """Return the value of the weight and bias initialisation for input x"""
        W = np.arange(-0.5 * size_in * size, 0.5 * size_in * size).reshape((size, size_in))
        b = np.arange(-0.5 * size, 0.5 * size).reshape((size, 1))

        return W, b

# --------------------------------------------------------------------------------------------
# Normal Gaussian distribution Weight and Bias Initialisation function
# Returns W (size, size_in) and b (size, 1)
# --------------------------------------------------------------------------------------------
class NormalWeightBiasInit(object):

    @staticmethod
    def fn(size_in, size):
        """Return the value of the weight and bias initialisation for input x"""
        W = np.random.randn(size, size_in)
        b = np.random.randn(size, 1)

        return W, b

# --------------------------------------------------------------------------------------------
# Dummy Activation function for testing
# --------------------------------------------------------------------------------------------
class TestActivation(object):

    @staticmethod
    def fn(x):
        """Return the value of the activation function for input x"""
        return x + 1

    @staticmethod
    def dfndx(y):
        """Return the derivative of the activation function"""
        return y - 9

# --------------------------------------------------------------------------------------------
# Sigmoid Activation function
# Consists of two functions: the first is used during forward prop and the second for back prop
# Returns matrices of the same shape as the input
# --------------------------------------------------------------------------------------------
class SigmoidActivation(object):

    @staticmethod
    def fn(x):
        """Return the value of the sigmoid function for input x"""
        return 1.0/(1.0+np.exp(-x))

    @staticmethod
    def dfndx(y):
        """Return the derivative of the sigmoid function"""
        return SigmoidActivation.fn(y)*(1-SigmoidActivation.fn(y))

# --------------------------------------------------------------------------------------------
# Quadratic Cost function based on Mean Square Error
# Consists of two functions: the first is used during forward prop and the second for back prop
# --------------------------------------------------------------------------------------------
class QuadraticCost(object):

    @staticmethod
    def fn(y, t):
        """Return the cost associated with an output ``y`` and desired output``t``.
        """
        return 0.5*np.linalg.norm(y-t)**2

    @staticmethod
    def dCdy(y, t):
        """Return the derivative of the cost"""
        return (y-t)

# --------------------------------------------------------------------------------------------
# Cross Entropy Cost function
# Consists of two functions: the first is used during forward prop and the second for back prop
# --------------------------------------------------------------------------------------------
class CrossEntropyCost(object):

    @staticmethod
    def fn(y, t):
        """Return the cost associated with an output ``y`` and desired output
        ``t``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``y`` and ``t`` have a 1.0
        in the same slot, then the expression (1-t)*np.log(1-y)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-t*np.log(y)-(1-t)*np.log(1-y)))

    @staticmethod
    def dCdy(y, t):
        """Return the derivative of the cost
        
        !!! THIS IS NOT CORRECT. PLUG IN THE CORRECT FORMULA HERE !!!! 

        """
        return (y-t)

# --------------------------------------------------------------------------------------------
# The Layer is the primary object in the network. The layer consists of several matrices
# which represent the nodes and related components in that layer. These matrices are the 
# input values, the intermediate values, the output values, the weights, the biases and the 
# various derivatives of the cost wrt each of these.
#
# The Layer performs three main operations:
#   Forward - computes the output value of the layer based on the input value. These will 
#                get passed on to the next layer. The output is computed in two steps. The
#                first step computes an intermediate value from the input, and the second
#                step computes the output value by applying an activation function to the
#                intermediate value
#   Backward - the next layer passes in the derivative of the cost wrt the output of this
#                layer. This layer in turn computes the derivatives of the cost wrt the 
#                intermediate values, wrt the weight, wrt the bias and wrt to the output
#                value of the previous layer, which will then get passed on to the prev layer.
#   Grad Descent - updates the weight and bias based on the derivatives (ie. gradients)
#                computed in the Backward step.
# --------------------------------------------------------------------------------------------
class Layer(object):
    def __init__(self, size_in, size, wbInit, activation):
        """ Initialize this layer with random weights and any other parameters we may need.
        """
        # size: the number of nodes in this layer
        # size_in: the number of nodes in the previous layer
        self.size = size

        # Weight and bias of the layer
        # W.shape = (size x size_in)
        # b.shape = (size x 1)
        self.W, self.b = wbInit.fn(size_in, size)

        # Y_in: output of the previous layer which is the input to this layer
        # X: Intermediate value of this layer computed from the input
        # Y: output of this layer (which becomes the input to the next layer)
        self.X = None # shape (size x 1)
        self.Y = None # shape (size x 1)
        self.Y_in = None # shape (size_in x 1)
        
        # dCdy: Derivative of Cost wrt output of this layer (which gets passed in from the next layer)
        # dCdx: Derivative of Cost wrt intermediate value of this layer
        self.dCdy = None # shape (size x 1)
        self.dCdx = None # shape (size x 1)

        # dCdW: Derivative of Cost wrt weight for one training sample
        # sigma_dCdW: Total Derivative of Cost wrt weight for all training samples
        # dCdb: as above for bias
        self.sigma_dCdW = np.zeros((size, size_in)) # shape (size x size_in)
        self.sigma_dCdb = np.zeros((size, 1)) # shape (size x 1)

        # Activation function to compute output value from the intermediate value
        self.activation = activation
        print ('Layer has size %d by %d' % (size_in, size))

    def reset_batch(self):
        """Reset the sigma gradient values before the start of the next batch 
        """
        self.sigma_dCdW = np.zeros(self.sigma_dCdW.shape)
        self.sigma_dCdb = np.zeros(self.sigma_dCdb.shape)

    def calc_X(self, Y_in):
        """Compute the X matrix given the input Y matrix from the previous layer, and 
        using the W and b matrices. 
        """
        self.X = np.dot(self.W, Y_in) + self.b

    def calc_Y(self):
        """Compute the Y matrix using the X matrix and the activation function 
        """
        self.Y = self.activation.fn(self.X)

    def calc_prevdCdy(self):
        """Compute the derivative matrix of the Cost to the output Y for the previous layer
        """
        prevdCdy = np.dot (self.W.transpose(), self.dCdx)
        return prevdCdy

    def calc_dCdx(self):
        """Compute the derivative matrix of the Cost to the input X
        """
        dydx = self.activation.dfndx(self.Y)
        self.dCdx = self.dCdy * dydx

    def calc_dCdW(self):
        """Compute the derivative matrix of the Cost to the weight W matrix
        """
        dxdW = self.Y_in.transpose()    # shape (1 x size_in)
        dCdW = np.dot (self.dCdx, dxdW) # shape (size x size_in)
        return dCdW

    def calc_dCdb(self):
        """Compute the derivative matrix of the Cost to the bias b matrix
        """
        m = self.dCdx.shape[1]
        ones = np.full ((self.dCdx.shape[1], 1), 1)
        dCdb = np.dot (self.dCdx, ones)   # shape (size x 1)
        return dCdb

    def sigma_dCdWdB(self, dCdW, dCdb):
        """Add the derivative matrix of Cost to weight and Cost to bias to the
        accumulated gradient from previous input samples
        """
        self.sigma_dCdW = self.sigma_dCdW + dCdW
        self.sigma_dCdb = self.sigma_dCdb + dCdb

    def forward(self, Y_in):
        """ Compute the output Y matrix from this layer, given the input Y matrix
        from the previous layer.
        """
        self.Y_in = Y_in
        self.calc_X(Y_in)
        self.calc_Y()
        return self.Y

    def backward(self, dCdy):
        """ Compute the gradient dCdW matrix from this layer, given the gradient dCdy matrix
        """
        self.dCdy = dCdy
        self.calc_dCdx()
        dCdW = self.calc_dCdW()
        dCdb = self.calc_dCdb()
        self.sigma_dCdWdB(dCdW, dCdb)

        # Now calculate the cost derivative relative to the Y of previous layer
        prevdCdy = self.calc_prevdCdy()
        return prevdCdy

    def gradDescent(self, learning_rate):
        # Perform the actual weight update step.
        print('Sigma weight gradient is ', self.sigma_dCdW)
        print('Sigma bias gradent is ', self.sigma_dCdb)

        batch_len = self.X.shape[1]
        self.W = self.W - ((learning_rate/batch_len) * self.sigma_dCdW)
        self.b = self.b - ((learning_rate/batch_len) * self.sigma_dCdb)
        #print('New weight is ', self.W)
        #print('New bias is ', self.b)

# --------------------------------------------------------------------------------------------
# The network consists simply of a sequence of layers. The goal of training the network is
# to find the optimum values of the Weights and the Biases in each layer, which can be
# thought of as the parameters of the network. The completed trained model basically means
# figuring out the best values of those Weights and Biases to be able to make accurate
# predictions. 
# 
# The network also has other hyperparameters which define the architecture of
# the network eg. the number of layers, number of nodes in each layer, how the weights are
# initialised, what activation function to use, what cost function to use, what learning
# rate to use and so on. Those constitute the basic structure of the network, so for a 
# particular training run those are fixed. However one can have multiple training runs where
# you experiment with different network architectures by trying different settings for those
# hyperparameters. That logic is not part of this implementation. This implementation does
# one training run. So you would call this implementation multiple times with different
# settings values for those hyperparameters. 
# 
# The training of the network then consists of feeding in the training data over and over 
# again. Each full pass over the data is called an epoch, and each epoch is broken down 
# into a number of mini-batches of data. During each mini-batch, the network processes 
# the input data in that mini-batch to compute the predicted output value based on that 
# input data. Those predicted values are then compared to the expected values via a cost 
# function (aka loss function). The objective is to reduce the cost from one mini-batch
# to the next by tuning the Weights and Biases. This is achieved via an optimisation
# algorithm called Stochastic Gradient Descent, which basically takes the derivatives
# of the cost function with respect to each of the Weights and Biases in the network
# after each mini-batch and then adjusts the values of those Weights and Biases by small
# amounts so as to reduce the loss. Those derivatives are essentially the Gradient
# of the cost.
# 
# The processing of a mini-batch consists of two flows, both of which proceed through 
# each layer one by one - a feed forward flow through each layer from left to right 
# and a backward propagation flow through each layer in the reverse direction from 
# right to left.
# 
# The purpose of the feed forward flow is to compute the predicted output values. The
# purpose of the back prop is to figure out the amounts by which the Weights and Biases
# should be adjusted. To do this, it has to compute the cost function, and then the
# derivative of that cost function with respect to the inputs, outputs, weights and biases
# of each layer, from right to left.
# --------------------------------------------------------------------------------------------
class Network(object):

    def __init__(self, layer_sizes, wbInit=TestWeightBiasInit, activation=TestActivation, cost=QuadraticCost):
        """The 

        """
        # Create a layer object for all layers except the input layer 
        self.layers = ([Layer(size_in, size, wbInit, activation) 
            for size_in, size in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.cost=cost

    def forward(self, X_in):
        # Feed forward computations for all layers in the network
        Y_in = X_in
        for layer in self.layers:
            Y_in = layer.forward(Y_in)
        
        print ('Output is ', Y_in)
        return Y_in

    def backward(self, Y_out, Y_targ):
        # Back propagation computations for all layers in the network
        dCdy = self.cost.dCdy(Y_out, Y_targ)
        for layer in reversed(self.layers):
            dCdy = layer.backward(dCdy)

    def mini_batch(self, X_in, Y_targ, learning_rate):
        Y_out = net.forward(X_in)
        net.backward(Y_out, Y_targ)
        for layer in self.layers:
            layer.gradDescent (learning_rate)
            layer.reset_batch()

    def train_model(self, X_in, Y_targ, epochs, batch_len, learning_rate):
        m = X_in.shape[1] # number of training samples

        for j in range(epochs):
            print ('This is epoch ', j)
            for k in range(0, m, batch_len):
                print ('This is batch ', k)
                self.mini_batch(X_in[:, k:k + batch_len], Y_targ[:, k:k + batch_len], learning_rate)
        """The 
            random.shuffle(X_in)
            mini_batches = [
                X_in[:, k:k + batch_len]
                for k in range(0, m, batch_len)]
            for mini_batch in mini_batches:
                self.mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))
        """

    def mini_batchSequential(self, m, X_in):
        for i in range(m):
            Y_out = net.forward(X_in[:, i].reshape((3,1)))
            net.backward(Y_out, Y_targ[:, i].reshape((2,1)))
 

#training_data, validation_data, test_data = load_data_wrapper()
m = 4
input = [2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1]
X_in = np.array(input).reshape((3,m))

target = [93, 78, 68, 93, 49, 60, 60, 49]
Y_targ = np.array(target).reshape((2,m))

net = Network([3, 4, 2])
net.train_model(X_in, Y_targ, 2, 2, 0.00001)
#net.mini_batch(X_in, Y_targ, 0.1)
#net = Network([784, 3, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#net.SGD(training_data, 3, 2, 3.0, test_data=test_data)