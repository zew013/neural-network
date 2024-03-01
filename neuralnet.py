from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
from numpy import exp

import util
from net_functions import *
from util import append_bias
from configuration import Config

from functools import reduce, partial


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.
    """

    def __init__(self, activation_type="sigmoid"):
        if activation_type not in self.types:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        self.forward, self.backward = self.types[activation_type]

    def __call__(self, z):
        return self.forward(z)

    types = {
        "sigmoid": (sigmoid, grad_sigmoid),
        "tanh": (tanh, grad_tanh),
        "ReLU": (ReLU, grad_ReLU),
        "output": (output, grad_output)
    }


class Regularization:

    def __init__(self, regularization, regularization_lambda):
        forward, backward = self.types[regularization]
        self.forward = partial(forward, l=regularization_lambda)
        self.backward = partial(backward, l=regularization_lambda)

    types = {
        "L1": (l1, grad_l1),
        "L2": (l2, grad_l2),
        False: (l_none, l_none)
    }


class Layer:
    """
    This class implements Fully Connected layers for your neural network.
    """

    w: np.ndarray
    v: np.ndarray
    x: np.ndarray
    a: np.ndarray
    z: np.ndarray
    dw: np.ndarray

    activation: Activation
    regularization = Regularization

    config: Config

    def __init__(self, layerIdx: int, config: Config):
        """
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        in_units, out_units = config.layer_specs[layerIdx], config.layer_specs[layerIdx + 1]

        self.layerIdx = layerIdx

        self.activation = Activation("output") \
            if layerIdx == len(config.layer_specs) - 2 \
            else Activation(config.activation)

        self.regularization = Regularization(config.regularization_type, config.regularization_lambda)

        if config.weight_type == 'random':
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        if config.momentum:
            self.v = np.zeros((in_units + 1, out_units))

        self.config = config

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.a = x @ self.w
        self.z = self.activation(self.a)
        
        return self.z

    def backward(self, deltaCur, gradReqd=True):
        """
       Write the code for backward pass. This takes in gradient from its next layer as
        input and computes gradient for its weights and the delta to pass to its previous layers. 
        gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        
        The delta expression (that you prove in PA1 part1) for any layer consists of delta and 
        weights from the next layer and derivative of the activation function of weighted 
        inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be 
        multiplied with the derivative of the activation function of the weighted input of the 
        current layer to actually get the delta for the current layer. Remember, this is just 
        one way of interpreting it and you are free to interpret it any other way.
        
        Feel free to change the function signature if you think of an alternative way to implement the 
        delta calculation or the backward pass
        """

        # temporarily don't implement regularization
        # if regularization[0] == 1:
        #     self.dw = -(x.T @ deltaCur + regularization[1] * self.w) / x.shape[0]
        # elif regularization[0] == 2:
        #     self.dw = -(x.T @ deltaCur + regularization[1]) / x.shape[0]
        # else:
        #     self.dw = -(x.T @ deltaCur) / x.shape[0]

        # x shape: (batch_size, in_units + 1), deltaCur shape: (batch_size, out_units)

        # if deltaCur.shape[-1] - self.w.shape[-1] == 1:
        #     deltaCur = deltaCur[:, :-1]

        # deltaCur came from the next layer, but g'(a) is for the current layer
        # we need to calculate g'(a) for the current layer and update deltaCur
        deltaCur = self.activation.backward(self.a) * deltaCur

        self.dw = -(self.x.T @ deltaCur / self.x.shape[0]) + self.regularization.backward(self.w)

        # self.dw -=

        # Before we update the weights, we calculate the delta to pass to the previous layer. With all the weights this
        # will give (batch_size, out_units) * (out_units, in_units + 1) = (batch_size, in_units + 1)
        # however, there are actually only in_units units, so we need to remove the last column which multiplies with a
        # 'fake' unit with input always 1 (artificially appended)
        deltaPrev = deltaCur @ self.w[:-1, :].T

        # if self.v:
        #     self.v = self.config.momentum_gamma * self.v - self.config.learning_rate * self.dw
        #     self.w += self.v
        # else:
        if gradReqd:
            if self.config.momentum:
                self.v = self.config.momentum_gamma * self.v - self.config.learning_rate * self.dw
                self.w += self.v
            else:
                self.w -= self.config.learning_rate * self.dw

        if self.layerIdx == 0:
            return None  # already last layer, no need to calculate new delta

        return deltaPrev

        # raise NotImplementedError("Backward propagation not implemented for Layer")

    # to nice string representation
    def __repr__(self):
        return f"Layer {self.layerIdx}: {self.w.shape[0] - 1} -> {self.w.shape[1]} ({self.activation.activation_type})"


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        TODO
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.num_layers = len(config.layer_specs) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets

        self.regularization = config.regularization_type

        self.layers = [Layer(i, config) for i in range(self.num_layers)]  # Store all layers in this list.

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None) -> Tuple[np.ndarray, Optional[util.ModelResult]]:
        """
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        # return reduce(lambda fst, snd: snd(fst), self.layers, x)
        # for layer in self.layers:
        #     x = append_bias(layer(x))  # after the mapping, add bias again

        output = reduce(lambda fst, snd: append_bias(snd(fst)), self.layers, x)[:, :-1]
        # strip out the extra bias for output

        # compute loss and accuracy
        if targets is not None:
            self.targets = targets
            self.y = output
            loss = self.loss(output, targets)

            # need to account for regularization
            # loss += sum(layer.regularization.forward(layer.w) for layer in self.layers)

            acc = util.accuracy(output, targets)
            return output, util.ModelResult(loss, acc)

        return output, None

    @staticmethod
    def loss(y: np.ndarray, t: np.ndarray) -> float:
        """
        compute the categorical cross-entropy loss and return it.
        """

        epsilon = 0

        # print("y", y.shape)
        # print("t", t.shape)

        return -(t * np.log(y + epsilon)).sum() / len(t)

    def backward(self, gradReqd=True):
        """
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        """

        currDelta = self.targets - self.y

        for layer in reversed(self.layers):
            currDelta = layer.backward(currDelta, gradReqd)

        # first, call backward on the last layer,

    def __repr__(self):
        # description of all layers
        return "{}".format(self.layers)
