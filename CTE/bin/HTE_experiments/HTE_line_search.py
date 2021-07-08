import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
import torch

def line_search():
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

    # choose between - NOF / NOBF / NOL / NODO
    # for the other parameters, write whatever you think fit
    tuning_parameter = 'NOF'
    experiment_number = 0

    # search parameters
    num_of_ferns = [20, 50, 70, 100, 150, 200, 300]
    number_of_BF = [7]*len(num_of_ferns)
    num_of_layers = 2

    # optimization Parameters
    args.num_of_epochs = 50
    args.batch_size = 250
    args.word_calc_learning_rate = 0.01
    args.voting_table_learning_rate = 0.01
    args.LR_decay = 0.99
    args.optimizer = 'ADAM' # ADAM / sgd
    args.loss = 'categorical_crossentropy'
    args.Rho_end_value = 0.3
    args.end_rho_at_epoch = args.num_of_epochs - 10
    args.batch_norm = True
    args.res_connection = 2 # 1 - resnet from input, size of d_out for l in [0, l-1] is d_in (summing input with layer's output)
                            # 2 - resnet concatination, d_out for l in [0, l-1] is concatanated with input features

    # create data-loaders
    dataset_name = 'LETTER' # LETTER / ADULT / Helena / Iris
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', dataset_name)
    args.datapath = os.path.join(args.datadir, 'split_data')

    dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 4}

    if dataset_name == 'LETTER':
        from CTE.utils.datasets.Letter_dataset import Letters as DataSet
        from CTE.utils.datasets.create_letters_dataset import main as create_dataset
        from CTE.bin.HTE_experiments.HTE_Letter_recognition import Train_Letters as Train_model
    elif dataset_name == 'ADULT':
        from CTE.utils.datasets.Adult_dataset import Adult_1_hot as DataSet
        from CTE.utils.datasets.create_adult_dataset import main as create_dataset
        from CTE.bin.HTE_experiments.HTE_Letter_recognition import Train_Letters as Train_model

    train_path, test_path = create_dataset(args)
    training_set = DataSet(train_path)
    train_loader = torch.utils.data.DataLoader(training_set, **dataset_params)
    testing_set = DataSet(test_path, train_loader.dataset.mean, train_loader.dataset.std)
    test_loader = torch.utils.data.DataLoader(testing_set, **dataset_params)

    args.debug = False # debugging the network using a pre-defined ferns and tables
    args.visu_progress = True # visualizing training graphs

    folder_to_save = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', dataset_name)

    for i in range(len(num_of_ferns)):

        args.experiment_name = 'exp ' + str(experiment_number) + " tuning " + tuning_parameter + ', ' + str(num_of_layers) + "_layers with " + str(num_of_ferns[i]) + ' ferns'
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]
        args.num_of_layers = num_of_layers
        # paths to save models
        args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', folder_to_save, args.experiment_name)
        os.makedirs(args.save_path, exist_ok=True)

        Train_model(args, train_loader, test_loader, device)

if __name__ == '__main__':
    line_search()
