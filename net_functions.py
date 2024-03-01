import numpy as np


def l1(w, l):
    return l * np.sum(np.abs(w))


def grad_l1(w, l):
    return l * np.sign(w)


def l2(w, l):
    return l * (w ** 2).sum() / 2


def grad_l2(w, l):
    return l * w


def l_none(w, l):
    return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def ReLU(x):
    return x * (x > 0)


def output(x):
    x -= np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def grad_sigmoid(x):
    g = sigmoid(x)
    return g * (1 - g)


def grad_tanh(x):
    return 1 - tanh(x) ** 2


def grad_ReLU(x):
    return (x > 0) * 1


def grad_output(x):
    return 1
