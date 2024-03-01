configYamlPath = "./configs/"  # Change it according to the directory of the platform on which the code is being run

datasetDir = "./data/"
cifar10_directory = "cifar-10-batches-py"
cifar10_trainBatchFiles = 5

saveLocation = "./plots/"
IMAGE_SIZE = 32


"""
    if args.experiment == 'test_gradients':  # 2b
        configFile = None  # Create a config file for 2b and change None to the config file name
    elif args.experiment == 'test_momentum':  # 2c
        configFile = "config_1.yaml"
    elif args.experiment == 'test_regularization':  # 2d
        configFile = None  # Create a config file for 2d and change None to the config file name
    elif args.experiment == 'test_activation':  # 2e
        configFile = None  # Create a config file for 2e and change None to the config file name
    elif args.experiment == 'test_hidden_units':  # 2f-i
        configFile = None  # Create a config file for 2f-i and change None to the config file name
    elif args.experiment == 'test_hidden_layers':  # 2f-ii
        configFile = None  # Create a config file for 2b and change None to the config file name
"""
EXPERIMENT_CONFIG_FILE = {
    # 'test_gradients': 'config_2b.yaml',
    # 'test_momentum': 'config_2c.yaml',
    #
    # 'test_regularization_L2_-2': '/config_2d/config_2d_L2_-2.yaml',
    # 'test_regularization_L2_-4': '/config_2d/config_2d_L2_-4.yaml',
    # 'test_regularization_L1_-2': '/config_2d/config_2d_L1_-2.yaml',
    # 'test_regularization_L1_-4': '/config_2d/config_2d_L1_-4.yaml',

    # 'test_activation_sigmoid': '/config_2e/config_2e_sigmoid.yaml',
    # 'test_activation_ReLU': '/config_2e/config_2e_ReLU.yaml',

    # 'test_hidden_units_half': '/config_2f/config_2f_half.yaml',
    # 'test_hidden_units_double': '/config_2f/config_2f_double.yaml',
    # 'test_hidden_units_thin': '/config_2f/config_2f_thin.yaml',



    # 'test_hidden_layers': 'config_2f_ii.yaml',
    'best': 'config_best.yaml'
}