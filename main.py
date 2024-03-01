################################################################################
# CSE 151B: Programming Assignment 2
# Fall 2022
# Code by Chaitanya Animesh & Shreyas Anantha Ramaprasad
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import gradient
from constants import *
from train import *
from gradient import *
import argparse
from configuration import load_config


# TODO
def main(args):
    # Read the required config
    # Create different config files for different experiments
    configFile = EXPERIMENT_CONFIG_FILE[args.experiment]
    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = load_config(configYamlPath + configFile)  # Set configYamlPath, configFile  in constants.py

    # Load the data
    train_dataset, val_dataset, test_dataset = util.load_data(path=datasetDir)  # Set datasetDir in constants.py

    if args.experiment == 'test_gradients':

        print(gradient.checkGradient(train_dataset, config))
        return 0

    model = Neuralnetwork(config)

    print(model)

    # train the model. Use train.py train method for this
    model, results_df = train(model, train_dataset, val_dataset, config)
    for col in ['accuracy', 'loss']:
        sns.lineplot(x='epoch', y=col, hue='type', data=results_df)
        # set title
        plt.title('Model {}'.format(col))

        # save as pdf name is the config name
        plt.savefig(f'{saveLocation}/{config.name}_{col}.pdf')
        plt.clf()

    # test the model. Use train.py modelTest method for this
    test_result = model_test(model, test_dataset)

    # append the test result to the results_df
    test_df = pd.DataFrame([test_result])
    test_df['type'] = 'test'
    test_df['epoch'] = -1
    results_df = pd.concat([results_df, test_df])

    # save the results_df as a csv
    results_df.to_csv(f'{saveLocation}/{config.name}_results.csv')

    # Print test accuracy and test loss
    print(test_result)


if __name__ == "__main__":
    # Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum',
                        help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)
