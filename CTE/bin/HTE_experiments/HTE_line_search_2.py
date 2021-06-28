import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
from CTE.bin.HTE_experiments.HTE_yeast import Train_Yeast
import torch
from CTE.utils.datasets import Yeast_dataset
from CTE.utils.datasets.create_yeast_dataset import main as create_dataset

def line_search():
    # search parameters
    num_of_ferns = [20, 50, 70, 100, 150, 200, 300, 400]
    number_of_BF = [8] * len(num_of_ferns)
    num_of_layers = 1

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # setting the device and verifying reproducibility
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(10)
    torch.manual_seed(0)
    args.device = device

    # optimization Parameters
    args.num_of_epochs = 50
    args.batch_size = 50
    args.word_calc_learning_rate = 0.01
    args.voting_table_learning_rate = 0.01
    args.LR_decay = 0.99
    args.optimizer = 'ADAM' # ADAM / sgd
    args.loss = 'categorical_crossentropy'
    args.batch_norm = True

    # create data-loaders
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'YEAST', 'yeast_zip', 'data')
    args.datapath = os.path.join(args.datadir, 'split_data')
    train_path, test_path = create_dataset(args)

    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
    training_set = Yeast_dataset.Yeast(train_path)
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    testing_set = Yeast_dataset.Yeast(test_path, train_loader.dataset.mean, train_loader.dataset.std)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)

    args.debug = False # debugging the network using a pre-defiend ferns and tables
    args.visu_progress = True # visualizing training graphs

    for i in range(len(num_of_ferns)):

        args.experiment_number = "line_search_Fern_" + str(num_of_layers) + "_layer_" + str(i)
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]
        args.num_of_layers = num_of_layers
        # paths to save models
        experiment_name = 'HTE-Yeast-resnet'
        experiment_number = args.experiment_number
        args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name, experiment_number)
        os.makedirs(args.save_path, exist_ok=True)

        Train_Yeast(args, train_loader, test_loader, device)

if __name__ == '__main__':
    line_search()
