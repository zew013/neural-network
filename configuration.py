import dataclasses
from typing import List, Literal, Optional

import yaml

"""
--- !Config
# This specifies the number of layers and number of hidden neurons in each layer.
# Note that the first and the last elements of the list indicate the input and
# output sizes
layer_specs: [3072, 128, 10]  #Represents a 2 layer NN. 3072 is the input layer

# Type of non-linear activation function to be used for the layers.
activation: "tanh"

# The learning rate to be used for training.
learning_rate: 0.01

# Number of training samples per batch to be passed to network
batch_size: 128

# Number of epochs to train the model
epochs: 100

# Flag to enable early stopping
early_stop: True

# History for early stopping. Wait for this many epochs to check validation loss / accuracy.
early_stop_epoch: 5


# Regularization
regularization_type: "L2"
# Regularization constant
regularization_lambda: 0.01 


# Use momentum for training
momentum: True
# Value for the parameter 'gamma' in momentum
momentum_gamma: 0.9

#Weight Type
weight_type: "random"
"""


@dataclasses.dataclass
class Config(yaml.YAMLObject):
    yaml_tag = u'!Config'

    layer_specs: List[int]
    activation: Literal["sigmoid", "tanh", "ReLU", "output"]

    learning_rate: float
    batch_size: int
    epochs: int

    early_stop: bool
    early_stop_epoch: Optional[int]

    regularization_type: Literal["L1", "L2", False]
    regularization_lambda: Optional[float]

    momentum: bool
    momentum_gamma: Optional[float]

    weight_type: str

    name: str = "default"


def load_config(path) -> Config:
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    res = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)

    res.name = path.split('/')[-1].split('.')[0]
    assert isinstance(res, Config), "Please add '--- !Config' to the top of the config file, otherwise it will not load"
    return res
