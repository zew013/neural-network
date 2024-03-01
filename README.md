# Image Classification with Multi-layer Perceptron

This is the code for the second PA of CSE151B. The goal of this PA is to implement the multiple-layer perceptron for image classification. 
The code is written in Python 3.10 and tested on Mac OS X 10.12.6 and Windows 10.
With default configuration, the code works pretty well with around 44% accuracy, and even better for manually choosing hyperparameters with around 0.4908.
## Code Structure

```
.
├── README.md # This file
├── cifar-10-batches-py # The CIFAR-10 dataset
├── get_cifar10data.sh # The script to download the CIFAR-10 dataset
├── main.py # The main program entry, train and test the network performance
├── neuralnet.py # The network implementation
├── net_functions.py # the implementation of helper function for network
├── out # The output directory for report
├── plots # The output directory for plots
├── report # The report latex source directory
├── train.py # The training function implementation
├── utils.py # The utility functions
├── configuration.py # The file for handling args to yaml files
├── constants.py # The file contains quick access for yaml files
├── driver.py # The implementation of inputting all yaml files at once
├── gradient.py # Check the correctness of gradient functions in the network.
```

## Usage
```shell
$ python ./main.py -h
usage: main.py [-h] [--experiment EXPERIMENT]

options:
  -h, --help            show this help message and exit
  --experiment EXPERIMENT
                        Specify the experiment that you want to run 
                        in the EXPERIMENT_CONFIG_FILE variable of constants.py

```
