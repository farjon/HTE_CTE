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

    args.results_csv_file = os.path.join(GetEnvVar('ExpResultsPath'), 'HTE', 'exp_results_master_file.csv')

    # choose between - NOF / NOBF / NOL / NODO
    # for the other parameters, write whatever you think fit
    tuning_parameter = 'NOF'
    args.experiment_number = 3

    # search parameters
    num_of_ferns = [100]
    number_of_BF = [8]
    num_of_layers = 2

    # optimization Parameters
    args.num_of_epochs = 70
    args.batch_size = 1024
    args.word_calc_learning_rate = 0.01
    args.voting_table_learning_rate = 0.01
    args.LR_decay = 0.99
    args.optimizer = 'ADAM' # ADAM / sgd
    args.loss = 'MSE' # MSE / uncertinty_yarin_gal
    args.use_cosine_lr = False
    args.weight_decay = 1e-2 # place 0 for no weight decay
    args.Rho_end_value = 0.3
    args.end_rho_at_epoch = args.num_of_epochs - 30
    args.batch_norm = True
    args.res_connection = 2 # 1 - resnet from input, size of d_out for l in [0, l-1] is d_in (summing input with layer's output)
                            # 2 - resnet concatination, d_out for l in [0, l-1] is concatanated with input features
    args.task = 'reg'
    args.use_mixup = False

    args.monitor_mse = True
    args.monitor_mae = False

    assert args.monitor_mse + args.monitor_mae == 1, 'you can monitor only a single metric'


    # create data-loaders
    args.dataset_name = 'year' # california_housing / year
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE Guy dataset', 'HTE_data', args.dataset_name)

    dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}

    if args.dataset_name == 'california_housing':
        from CTE.bin.HTE_experiments.HTE_Cali_Housing import Train_Cali_Housing as Train_model
    elif args.dataset_name == 'year':
        from CTE.bin.HTE_experiments.HTE_Year import Train_Year as Train_model

    from CTE.utils.datasets.Tabular_dataset import TabularDataset as DataSet

    training_set = DataSet(args.datadir, set='train', device=device)
    train_loader = torch.utils.data.DataLoader(training_set, **dataset_params)
    testing_set = DataSet(args.datadir, set='test', device=device)
    test_loader = torch.utils.data.DataLoader(testing_set, **dataset_params)

    validation_set = DataSet(args.datadir, set='val', device=device)
    val_loader = torch.utils.data.DataLoader(validation_set, **dataset_params)

    args.debug = False # debugging the network using a pre-defined ferns and tables
    args.visu_progress = True # visualizing training graphs

    folder_to_save = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', args.dataset_name)

    for i in range(len(num_of_ferns)):
        print(f'this is experiment number {args.experiment_number} now running the {i}th loop')
        args.experiment_name = 'exp ' + str(args.experiment_number) + " tuning " + tuning_parameter + ', ' + str(num_of_layers) + "_layers with " + str(number_of_BF[i]) + ' BF'
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]
        args.num_of_layers = num_of_layers
        # paths to save models
        args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', folder_to_save, args.experiment_name)
        os.makedirs(args.save_path, exist_ok=True)
        Train_model(args, train_loader, test_loader, device, val_loader)

if __name__ == '__main__':
    line_search()
