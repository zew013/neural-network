import numpy as np
from neuralnet import Neuralnetwork
from util import Dataset
import pandas as pd


def check_grad(model, x_train, y_train):
    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-2
    model.forward(x_train, y_train)
    model.backward(False)

    classes = {'output bias weight': (model.layers[-1], -1, 2),  # one output bias weight
               'hidden bias weight': (model.layers[-2], -1, 2),  # one hidden bias weight
               'hidden to output 1': (model.layers[-1], -2, 2),  # hidden to output
               'hidden to output 2': (model.layers[-1], -3, 2),
               'input to hidden 1': (model.layers[0], -3, 2),  # input to hidden
               'input to hidden 2': (model.layers[0], -4, 2)
               }
    results = []
    for key, tup in classes.items():
        out_layer, layer_position, layer_element = tup

        out_layer.w[layer_position][layer_element] += epsilon
        Ew_plus = model.forward(x_train, y_train)[1].loss
        out_layer.w[layer_position][layer_element] -= 2 * epsilon
        Ew_minus = model.forward(x_train, y_train)[1].loss

        dw_output_bias1 = (Ew_plus - Ew_minus) / (2 * epsilon)
        dw_output_bias0 = out_layer.dw[layer_position][layer_element]

        results.append([key, dw_output_bias1, dw_output_bias0, abs(dw_output_bias0 - dw_output_bias1)])
    return pd.DataFrame(results, columns=['Types of Weight', 'Gradient Approxi', 'Gradient Backprop', 'Abs Diff'])

    # print(model.layers[-2].w[-1][0])

    # print(model.layers[-1].w[-2][0])

    # print(model.layers[-1].w[-3][2])


def checkGradient(dataset: Dataset, config):
    subsetSize = 100  # Feel free to change this
    sample_idx = np.random.randint(0, len(dataset.X), subsetSize)
    x_train_sample, y_train_sample = dataset.X[sample_idx], dataset.t[sample_idx]

    model = Neuralnetwork(config)
    return check_grad(model, x_train_sample, y_train_sample)
