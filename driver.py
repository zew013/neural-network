import argparse

from constants import EXPERIMENT_CONFIG_FILE
from main import main

if __name__ == '__main__':
    for exp in EXPERIMENT_CONFIG_FILE.keys():
        print(f'Running experiment {exp}')
        main(argparse.Namespace(experiment=exp))