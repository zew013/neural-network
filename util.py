import os
from dataclasses import dataclass
from typing import Tuple, Iterator, NamedTuple

import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants
from configuration import Config
from constants import IMAGE_SIZE


@dataclass
class Dataset:
    """
    Class for encapsulating iamges and labels
    """
    X: np.ndarray
    t: np.ndarray


@dataclass
class ModelResult:
    loss: float
    accuracy: float

    # override the __add__ method to add two ModelResult objects
    def __add__(self, other):
        return ModelResult(self.loss + other.loss, self.accuracy + other.accuracy)

    # override the __truediv__ method to divide a ModelResult object by a scalar
    def __truediv__(self, other):
        return ModelResult(self.loss / other, self.accuracy / other)

    # override the __mul__ method to multiply a ModelResult object by a scalar
    def __mul__(self, other):
        return ModelResult(self.loss * other, self.accuracy * other)

    # compare based on loss
    def __lt__(self, other):
        return self.loss < other.loss


def normalize_data(inp: np.ndarray) -> np.ndarray:
    """
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to separate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """

    # reshape to separate channels
    inp = inp.reshape((-1, 3, IMAGE_SIZE, IMAGE_SIZE))  # shape = (N, C, H, W)

    # normalize

    mean = np.mean(inp, axis=(0, 2, 3), keepdims=True)  # along dimensions (N, H, W), i,e, per channel
    std = np.std(inp, axis=(0, 2, 3), keepdims=True)
    inp = (inp - mean) / std

    # reshape back
    inp = inp.reshape((-1, IMAGE_SIZE * IMAGE_SIZE * 3))
    return inp


def onehot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (10 for CIFAR-10)

    returns:
        oneHot : N X num_classes 2D array

    """

    return np.eye(num_classes)[labels]


def onehot_decode(t: np.ndarray) -> np.ndarray:
    """
    Decodes the one hot encoded labels
    """
    return np.argmax(t, axis=1)


def generate_minibatches(dataset: Dataset, batch_size=64) -> Iterator[Dataset]:
    """
        Generates minibatches of the dataset

        args:
            dataset: Dataset object
            batch_size: size of the minibatch
        yields:
            minibatch: Dataset object

        """

    X, t = dataset.X, dataset.t
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield Dataset(X[l_idx:r_idx], t[l_idx:r_idx])
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield Dataset(X[l_idx:], t[l_idx:])


def accuracy(y, t) -> float:
    """
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        accuracy: Accuracy of the predictions
    """

    return (onehot_decode(y) == onehot_decode(t)).mean()


def append_bias(X: np.ndarray) -> np.ndarray:
    """
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)


def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):
    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. epoch=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1, len(trainEpochLoss) + 1, 1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop], valEpochLoss[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation + "loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation + "accuarcy.eps")
    plt.show()

    # Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation + "trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation + "valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation + "trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation + "valEpochAccuracy.csv")


def shuffle(dataset: Dataset) -> Dataset:
    p = np.random.permutation(len(dataset.X))
    return Dataset(dataset.X[p], dataset.t[p])


def split_train_val(dataset: Dataset, valSplit: float = 0.2) -> Tuple[Dataset, Dataset]:
    """
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """

    # shuffle
    dataset = shuffle(dataset)

    # split
    split = int(len(dataset.X) * (1 - valSplit))
    return Dataset(dataset.X[:split], dataset.t[:split]), Dataset(dataset.X[split:], dataset.t[split:])


def preprocess(dataset: Dataset) -> Dataset:
    """
    Preprocess the dataset (normalize X and onehot encode t)
    """

    return Dataset(append_bias(normalize_data(dataset.X)), onehot_encode(dataset.t))


def load_data(path) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads, splits our dataset- CIFAR-10 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-10 dataset
    returns:
        train, val, test datasets
    """

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar10_directory)

    train_images = []
    train_labels = []
    for i in range(1, constants.cifar10_trainBatchFiles + 1):
        images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
        data = images_dict[b'data']
        label = images_dict[b'labels']
        train_labels.extend(label)
        train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape(len(train_labels))  # reshape to (N,)

    train_dataset = Dataset(train_images, train_labels)

    train_dataset, val_dataset = map(preprocess, split_train_val(train_dataset))

    test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape(len(test_labels))

    test_dataset = Dataset(test_images, test_labels)
    test_dataset = preprocess(test_dataset)

    return train_dataset, val_dataset, test_dataset
