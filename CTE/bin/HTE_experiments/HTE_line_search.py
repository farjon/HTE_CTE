import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
import torch

def line_search():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # setting the device and verifying reproducibility
    # device = torch.device('cpu')
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
    args.experiment_number = 13

    # search parameters
    num_of_ferns = [100]
    number_of_BF = [7]*len(num_of_ferns)
    num_of_layers = 3

    # optimization Parameters
    args.num_of_epochs = 160
    args.batch_size = 256
    args.word_calc_learning_rate = 0.01
    args.voting_table_learning_rate = 0.01
    args.LR_decay = 0.995
    args.optimizer = 'ADAM' # ADAM / sgd
    args.loss = 'categorical_crossentropy'
    args.use_cosine_lr = False
    args.weight_decay = 1e-2 # place 0 for no weight decay
    args.Rho_end_value = 0.2
    args.end_rho_at_epoch = args.num_of_epochs - 20
    args.batch_norm = True
    args.res_connection = 2 # 1 - resnet from input, size of d_out for l in [0, l-1] is d_in (summing input with layer's output)
                            # 2 - resnet concatination, d_out for l in [0, l-1] is concatanated with input features

    args.use_mixup = False

    args.monitor_acc = False
    args.monitor_balanced_acc = True
    args.monitor_auc = False

    assert args.monitor_acc + args.monitor_balanced_acc + args.monitor_auc == 1, 'you can monitor only a single metric'

    # create data-loaders
    args.dataset_name = 'adult' # LETTER / adult / higgs_small / aloi / helena / jannis
    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE Guy dataset', 'HTE_data', args.dataset_name)
    args.datapath = os.path.join(args.datadir)

    dataset_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}

    if args.dataset_name == 'LETTER':
        from CTE.utils.datasets.Letter_dataset import Letters as DataSet
        from CTE.utils.datasets.create_letters_dataset import main as create_dataset
        from CTE.bin.HTE_experiments.HTE_Letter_recognition import Train_Letters as Train_model
        train_path, val_path, test_path = create_dataset(args)
    elif args.dataset_name == 'adult':
        from CTE.utils.datasets.Adult_dataset import Adult as DataSet
        from CTE.bin.HTE_experiments.HTE_Adult import Train_Adult as Train_model
        train_path, test_path, val_path = 'train', 'test', 'val'
    elif args.dataset_name == 'higgs_small':
        from CTE.utils.datasets.Higgs_Small_dataset import Higgs_Small as DataSet
        from CTE.bin.HTE_experiments.HTE_Higgs import Train_Higgs as Train_model
        train_path, test_path, val_path = 'train', 'test', 'val'
    elif args.dataset_name == 'aloi':
        from CTE.utils.datasets.ALOI_dataset import ALOI as DataSet
        from CTE.bin.HTE_experiments.HTE_ALOI import Train_ALOI as Train_model
        train_path, test_path, val_path = 'train', 'test', 'val'
    elif args.dataset_name == 'helena':
        from CTE.utils.datasets.Helena_dataset import Helena as DataSet
        from CTE.bin.HTE_experiments.HTE_Helena import Train_Helena as Train_model
        train_path, test_path, val_path = 'train', 'test', 'val'
    elif args.dataset_name == 'jannis':
        from CTE.utils.datasets.Jannis_dataset import Jannis as DataSet
        from CTE.bin.HTE_experiments.HTE_Jannis import Train_Jannis as Train_model
        train_path, test_path, val_path = 'train', 'test', 'val'


    # training_set = DataSet(train_path)
    # train_loader = torch.utils.data.DataLoader(training_set, **dataset_params)
    # testing_set = DataSet(test_path, train_loader.dataset.mean, train_loader.dataset.std)
    # test_loader = torch.utils.data.DataLoader(testing_set, **dataset_params)
    #
    # if 'val_path' in locals():
    #     validation_set = DataSet(val_path, train_loader.dataset.mean, train_loader.dataset.std)
    #     val_loader = torch.utils.data.DataLoader(validation_set, **dataset_params)

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
        args.experiment_name = 'exp ' + str(args.experiment_number) + " tuning " + tuning_parameter + ', ' + str(num_of_layers) + "_layers with " + str(num_of_ferns[i]) + ' ferns'
        args.num_of_ferns = num_of_ferns[i]
        args.number_of_BF = number_of_BF[i]
        args.num_of_layers = num_of_layers
        # paths to save models
        args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', folder_to_save, args.experiment_name)
        os.makedirs(args.save_path, exist_ok=True)
        if 'val_loader' in locals():
            Train_model(args, train_loader, test_loader, device, val_loader)
        else:
            Train_model(args, train_loader, test_loader, device)

if __name__ == '__main__':
    line_search()
