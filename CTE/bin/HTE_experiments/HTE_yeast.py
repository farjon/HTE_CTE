import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
from CTE.models.HTE_model import HTE
import torch
import torch.nn as nn
from CTE.utils.datasets import Yeast_dataset
from torch import optim
from CTE.utils.help_funcs import save_anneal_params, load_anneal_params, print_end_experiment_report
from CTE.bin.HTE_experiments.training_functions import train_loop
from CTE.utils.datasets.create_yeast_dataset import main as create_yeast_dataset

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(10)
    torch.manual_seed(0)

    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="HTE model")
    args = parser.parse_args()

    #debug_mode - also used to visualize and follow specific parameters. See ../CTE/utils/visualize_function.py
    args.debug = True
    args.visu_progress = True
    args.draw_line = True

    # path to save models
    experiment_name = 'HTE-Yeast'
    experiment_number = '1'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name,
                                        experiment_number)
    args.path_to_parameters_save = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name,
                                        experiment_number, 'parameters_final_values.csv')

    # optimization Parameters
    args.word_calc_learning_rate = 0.001
    args.voting_table_learning_rate = 0.01

    args.LR_decay = 0.999
    args.num_of_epochs = 1
    args.batch_size = 50
    args.optimizer = 'ADAM'
    args.loss = 'categorical_crossentropy'

    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'YEAST', 'yeast_zip', 'data')

    args.datapath = os.path.join(args.datadir, 'split_data')
    train_path, val_path, test_path = create_yeast_dataset(args)

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 0}

    # create train,val,test data_loader
    training_set = Yeast_dataset.Yeast(train_path)
    train_loader = torch.utils.data.DataLoader(training_set, **params)

    train_mean = train_loader.dataset.mean
    train_std = train_loader.dataset.std

    validation_set = Yeast_dataset.Yeast(val_path, train_mean, train_std)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params)

    testing_set = Yeast_dataset.Yeast(test_path, train_mean, train_std)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)

    # Yeast dataset has 8 features
    D_in = 8
    D_out_1 = 15
    D_out = 10
    args.input_size = [args.batch_size, D_in]
    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern_layer = [
        {'K': 10, 'M': 10, 'num_of_features': D_in},
        {'K': 10, 'M': 10, 'num_of_features': D_out_1}
    ]
    args.ST_layer = [
        {'Num_of_active_words': 2**args.Fern_layer[0]['K'], 'D_out': D_out_1},
        {'Num_of_active_words': 2**args.Fern_layer[0]['K'], 'D_out': D_out}

    ]

    args.prune_type = 1
    args.number_of_layers = len(args.Fern_layer)

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    model = HTE(args, args.input_size, device)

    voting_table_LR_params_list = ['voting_table_layers.0.weights', 'voting_table.layers.0.bias', 'voting_table_layers.1.weights', 'voting_table.layers.1.bias']
    voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
    word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))

    optimizer = optim.Adam([{'params': word_calc_params},
                           {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                              lr=args.word_calc_learning_rate)

    saving_path = os.path.join(args.save_path)

    paths_to_save_anneal_params = []
    paths_to_save_anneal_params.append(os.path.join(saving_path, 'ambiguity_thresholds_layer_1.p'))
    paths_to_save_anneal_params.append(os.path.join(saving_path, 'anneal_params_1.p'))
    paths_to_save_anneal_params.append(os.path.join(saving_path, 'ambiguity_thresholds_layer_2.p'))
    paths_to_save_anneal_params.append(os.path.join(saving_path, 'anneal_params_2.p'))
    args.paths_to_save = paths_to_save_anneal_params

    def save_model_anneal_params(model, paths_to_save):
        # save first layer
        path_to_AT = paths_to_save[0]
        path_to_anneal_params = paths_to_save[1]
        save_anneal_params(model.word_calc_layers[0], path_to_AT, path_to_anneal_params)
        # save second layer
        path_to_AT = paths_to_save[2]
        path_to_anneal_params = paths_to_save[3]
        save_anneal_params(model.word_calc_layers[1], path_to_AT, path_to_anneal_params)

    def load_model_anneal_params(model, paths_to_save):
        # load first layer
        path_to_AT = paths_to_save[0]
        path_to_anneal_params = paths_to_save[1]
        # AT - ambiguity threshold, AP - anneal params
        AT, AP = load_anneal_params(path_to_AT, path_to_anneal_params)
        model.word_calc_layers[0].ambiguity_thresholds = AT
        model.word_calc_layers[0].anneal_state_params = AP
        # load second layer
        path_to_AT = paths_to_save[2]
        path_to_anneal_params = paths_to_save[3]
        # AT - ambiguity threshold, AP - anneal params
        AT, AP = load_anneal_params(path_to_AT, path_to_anneal_params)
        model.word_calc_layers[1].ambiguity_thresholds = AT
        model.word_calc_layers[1].anneal_state_params = AP
        return model

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    final_model = train_loop(args, train_loader, validation_loader, model, optimizer, criterion, device, saving_path, save_model_anneal_params)

    #save model and ambiguity_thresholds
    torch.save(final_model.state_dict(), os.path.join(saving_path, 'final_model_parameters.pth'))
    final_paths = []
    final_paths.append(os.path.join(saving_path, 'final_ambiguity_thresholds_layer_1.p'))
    final_paths.append(os.path.join(saving_path, 'final_anneal_params_1.p'))
    final_paths.append(os.path.join(saving_path, 'final_ambiguity_thresholds_layer_2.p'))
    final_paths.append(os.path.join(saving_path, 'final_anneal_params_2.p'))
    save_model_anneal_params(final_model, final_paths)

    test_model = HTE(args, args.input_size, device)
    test_model.load_state_dict(torch.load(os.path.join(saving_path, 'best_model_parameters.pth')), strict=False)
    test_model = load_model_anneal_params(test_model, args.paths_to_save)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = test_model(inputs_test)
            _, predicted = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()

            # print statistics
        print('Accuracy of the network on the %d test examples: %.2f %%' % (
        test_loader.dataset.examples.shape[0],
        100 * correct / total))

    print('Finished Training')
    print_end_experiment_report(args, model)


if __name__ == '__main__':
    main()

