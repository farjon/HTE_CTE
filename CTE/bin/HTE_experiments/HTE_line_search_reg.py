import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
import torch

def line_search():
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

    # choose between - NOF / NOBF / NOL / NODO
    # for the other parameters, write whatever you think fit
    tuning_parameter = 'NOBF'
    experiment_number = 1

    # search parameters
    num_of_ferns = [50]
    number_of_BF = [6]
    num_of_layers = 2

    # optimization Parameters
    args.num_of_epochs = 80
    args.batch_size = 400
    args.word_calc_learning_rate = 0.1
    args.voting_table_learning_rate = 0.01
    args.LR_decay = 0.99
    args.optimizer = 'ADAM' # ADAM / sgd
    args.loss = 'MSE' # MSE / uncertinty_yarin_gal
    args.Rho_end_value = 0.3
    args.end_rho_at_epoch = args.num_of_epochs - 30
    args.batch_norm = True
    args.res_connection = 2 # 1 - resnet from input, size of d_out for l in [0, l-1] is d_in (summing input with layer's output)
                            # 2 - resnet concatination, d_out for l in [0, l-1] is concatanated with input features

    # create data-loaders
    dataset_name = 'california_housing' # california_housing
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE Guy dataset', 'HTE_data', dataset_name)

    dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}

    if dataset_name == 'california_housing':
        from CTE.utils.datasets.CaliHousing import CaliHousing as DataSet
        from CTE.bin.HTE_experiments.HTE_Regression_exp import Train_Cali_Housing as Train_model

    training_set = DataSet(args.datadir, set='train', device=device)
    train_loader = torch.utils.data.DataLoader(training_set, **dataset_params)
    testing_set = DataSet(args.datadir, set='test', device=device)
    test_loader = torch.utils.data.DataLoader(testing_set, **dataset_params)

    validation_set = DataSet(args.datadir, set='val', device=device)
    val_loader = torch.utils.data.DataLoader(validation_set, **dataset_params)

    args.debug = False # debugging the network using a pre-defined ferns and tables
    args.visu_progress = True # visualizing training graphs

    folder_to_save = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', dataset_name)

    for i in range(len(num_of_ferns)):
        print(f'this is experiment number {experiment_number} now running the {i}th loop')
        args.experiment_name = 'exp ' + str(experiment_number) + " tuning " + tuning_parameter + ', ' + str(num_of_layers) + "_layers with " + str(number_of_BF[i]) + ' BF'
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]
        args.num_of_layers = num_of_layers
        # paths to save models
        args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', folder_to_save, args.experiment_name)
        os.makedirs(args.save_path, exist_ok=True)
        Train_model(args, train_loader, test_loader, device, val_loader)

if __name__ == '__main__':
    line_search()