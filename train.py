import copy
from typing import Tuple

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from configuration import Config
from neuralnet import *
from neuralnet import Neuralnetwork
from util import Dataset, shuffle, generate_minibatches, ModelResult

from tqdm import tqdm

import seaborn as sns


def train(model: Neuralnetwork, train_dataset: Dataset, val_dataset: Dataset, config: Config) -> Tuple[Neuralnetwork, pd.DataFrame]:
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        train_dataset - the training set
        val_dataset - the validation set
        config - the config object containing the parameters for training

    returns:
        the trained model
    """
    # Number of epochs to train the model
    epochs = config.epochs

    best_val_result = ModelResult(float('inf'), 0)
    best_model = copy.deepcopy(model)
    early_stop_epoch = config.early_stop_epoch

    results_df = pd.DataFrame(columns=['type', 'train', 'loss'])
    train_results = []
    val_results = []

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            # Train the model
            train_result = sgd(model, train_dataset, config)

            # Calculate the loss and accuracy on the validation set
            _, val_result = model.forward(val_dataset.X, val_dataset.t)

            # determine if early stopping is enabled
            if config.early_stop:

                if val_result.loss < best_val_result.loss:
                    best_val_result = val_result
                    best_model = copy.deepcopy(model)
                    early_stop_epoch = config.early_stop_epoch
                else:
                    early_stop_epoch -= 1
                    if not early_stop_epoch:
                        # early stopping
                        break

                train_results.append(train_result)
                val_results.append(val_result)

            pbar.set_postfix(train=train_result, val=val_result)
            pbar.update(1)

    # create two data frames, and concatenate them, then set the index to be the epoch
    train_df = pd.DataFrame(train_results)
    train_df['type'] = 'train'
    train_df['epoch'] = train_df.index
    val_df = pd.DataFrame(val_results)
    val_df['type'] = 'val'
    val_df['epoch'] = val_df.index
    results_df = pd.concat([train_df, val_df]).reset_index()



    return best_model, results_df


def sgd(model: Neuralnetwork, dataset: Dataset, config: Config) -> ModelResult:
    """
    Implements mini-batch SGD to train the model.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        dataset - the training set
        config - the config object containing the parameters for training

    returns:
        None
    """
    shuffle(dataset)

    batches = generate_minibatches(dataset, config.batch_size)

    mean_res = ModelResult(0, 0)

    for batch in batches:
        _, forward_res = model.forward(batch.X, batch.t)
        model.backward()

        batch_size = batch.X.shape[0]
        mean_res += forward_res * batch_size

    mean_res = mean_res / dataset.X.shape[0]
    return mean_res


# This is the test method
def model_test(model: Neuralnetwork, test_dataset: Dataset) -> ModelResult:
    """
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        test_dataset - the test set

    returns:
        the accuracy and loss on the test set
    """
    _, test_result = model.forward(test_dataset.X, test_dataset.t)
    return test_result
