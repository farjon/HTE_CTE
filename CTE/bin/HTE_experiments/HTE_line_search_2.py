import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
from CTE.models.HTE_ResFern import HTE
from CTE.bin.HTE_experiments.HTE_Letter_recognition import main
import torch
import torch.nn as nn
from CTE.utils.datasets import Letter_dataset
from torch import optim
from CTE.utils.help_funcs import save_anneal_params, load_anneal_params, print_end_experiment_report
from CTE.bin.HTE_experiments.training_functions import train_loop
from CTE.utils.datasets.create_letters_dataset import main as create_letters_dataset

def line_search():
    search_param = 1

    num_of_ferns = [10, 10, 70, 100, 150, 200]
    number_of_BF = [3, 3, 7, 7, 7, 7]


    for i in range(len(num_of_ferns)):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.experiment_number = "line_search_1_layer_" + str(i)
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]

        main(args)
        del args

if __name__ == '__main__':
    line_search()
