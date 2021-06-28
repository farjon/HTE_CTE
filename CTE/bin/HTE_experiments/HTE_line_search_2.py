import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
from CTE.models.HTE_ResFern import HTE
from CTE.bin.HTE_experiments.HTE_Letter_recognition import Train_Letters
import torch
import torch.nn as nn
from CTE.utils.datasets import Letter_dataset
from torch import optim
from CTE.utils.help_funcs import save_anneal_params, load_anneal_params, print_end_experiment_report
from CTE.bin.HTE_experiments.training_functions import train_loop
from CTE.utils.datasets.create_letters_dataset import main as create_letters_dataset

def line_search():
    # search parameters
    num_of_ferns = [20, 50, 70, 100, 150, 200, 300]
    number_of_BF = [8, 8, 8, 8, 8, 8, 8]
    num_of_layers = 3

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # setting the device and verifying reproducibility
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(10)
    torch.manual_seed(0)
    args.device = device

    # optimization Parameters
    args.num_of_epochs = 50
    args.batch_size = 400
    args.word_calc_learning_rate = 0.01
    args.voting_table_learning_rate = 0.01
    args.LR_decay = 0.99
    args.optimizer = 'ADAM' # ADAM / sgd
    args.loss = 'categorical_crossentropy'
    args.batch_norm = True

    # create data-loaders
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'LETTER')
    args.datapath = os.path.join(args.datadir, 'split_data')
    train_path, test_path = create_letters_dataset(args)

    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    training_set = Letter_dataset.Letters(train_path)
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    testing_set = Letter_dataset.Letters(test_path, train_loader.dataset.mean, train_loader.dataset.std)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)

    args.debug = False # debugging the network using a pre-defiend ferns and tables
    args.visu_progress = True # visualizing training graphs

    for i in range(len(num_of_ferns)):

        args.experiment_number = "line_search_Fern_" + str(num_of_layers) + "_layer_" + str(i)
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]
        args.num_of_layers = num_of_layers
        # paths to save models
        experiment_name = 'HTE-Letter-Recognition-resnet'
        experiment_number = args.experiment_number
        args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name, experiment_number)
        os.makedirs(args.save_path, exist_ok=True)

        Train_Letters(args, train_loader, test_loader, device)

if __name__ == '__main__':
    line_search()
