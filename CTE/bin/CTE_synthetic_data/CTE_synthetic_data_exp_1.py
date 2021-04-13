import numpy as np
import argparse
import os
import time
import sys
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.CTE_single_layer import CTE
from Sandboxes.Farjon.CTE.utils.CTE_Squares_Circles_dataset import CS_Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from Sandboxes.Farjon.CTE.utils.help_funcs import to_numpy, save_anneal_params, load_anneal_params
from Sandboxes.Farjon.CTE.utils.CTEOptimizer import CTEOptimizer
from itertools import count
import matplotlib.pyplot as plt


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(10)
    torch.manual_seed(10)


    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="CTE model")
    args = parser.parse_args()

    # create opts struct for all optimization parameters
    opts_parser = argparse.ArgumentParser(description="optimization parameters")
    opts = opts_parser.parse_args()

    #debug_mode - also used to visualize and follow specific parameters. See ../CTE/utils/visualize_function.py
    args.debug = True

    # path to save models
    experiment_name = 'Squares_and_Circles'
    experiment_number = '1'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name,
                                        experiment_number)

    # init_weights_base_dir = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', 'init ferns and S2D')

    args.layer1_fern_path = None
    args.layer1_S2D_path = None

    # optimization Parameters
    args.word_calc_learning_rate = 0.001

    args.voting_table_learning_rate = 0.1

    args.LR_decay = 0.99
    args.num_of_epochs = 150
    args.batch_size = 20
    args.loss = 'categorical_crossentropy'

    datapath_train = os.path.join(GetEnvVar('DatasetsPath'), 'CTE_synthetic data', 'Exp 1', 'train')
    datapath_test = os.path.join(GetEnvVar('DatasetsPath'), 'CTE_synthetic data', 'Exp 1', 'test')

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 0}

    training_set = CS_Dataset(datapath_train)
    train_loader = torch.utils.data.DataLoader(training_set, **params)

    testing_set = CS_Dataset(datapath_test)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)

    H = 50
    W = 50
    C = 1
    args.input_size = (H, W, C)

    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern1 = {'K': 4, 'M': 4, 'L': 19}
    args.ST1 = {'Num_of_active_words': 2**args.Fern1['K'], 'D_out': 2}
    args.AvgPool1 = {'kernel_size' : 32}

    args.prune_type = 1

    args.number_of_layer = 1

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')


        model = CTE(args, [H, W, args.batch_size], device)
        model.to(device)

        voting_table_LR_params_list = ['voting_table1.weights', 'voting_table2.weights', 'voting_table1.bias', 'voting_table2.bias']
        voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
        word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))

        optimizer = optim.Adam([{'params': word_calc_params},
                               {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                                  lr=args.word_calc_learning_rate)

        saving_path = args.save_path
        args.layer1_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_1.npy')
        args.layer1_word_calc_anneal_params = os.path.join(saving_path, 'first_layer_anneal_params.p')

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        best_accuracy = 0
        model.to(device)

        # plotting variables
        accuracy_graph = []
        loss_for_epoch =[]
        x_values = []
        index = count()

        for epoch in range(args.num_of_epochs):
            i = 0
            running_loss_print = 0.0
            running_loss_graph = 0.0
            for inputs, labels in train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs = torch.round(inputs)
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss_print += float(loss.item())
                running_loss_graph += float(loss.item())
                # print statistics
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss_print / 10))
                    running_loss_print = 0.0
                i = i + 1
                # on_batch_ends - callback
                model.on_batch_ends()
                optimizer.defaults['lr'] = optimizer.defaults['lr'] * args.LR_decay
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            if args.debug:
                accuracy_graph.append((correct_val / total_val))
                loss_for_epoch.append((running_loss_graph/len(train_loader.dataset.files_names)))
                x_values.append(next(index))
                plt.plot(x_values, accuracy_graph, label='accuracy')
                plt.plot(x_values, loss_for_epoch, label = 'loss')
                plt.legend()
                plt.xlim(0, args.num_of_epochs+4)
                plt.ylim(0, 1)
                plt.title('Experiment 1 Accuracy over epochs')
                plt.draw()
                if epoch == args.num_of_epochs-1:
                    plt.savefig(os.path.join(saving_path, 'progress plot.png'))
                plt.pause(.3)
                plt.cla()


            if best_accuracy <= (correct_val / total_val):
                torch.save(model.state_dict(), os.path.join(saving_path, 'best_model_parameters.pth'))
                best_accuracy = correct_val/total_val

            print('Accuracy of epoch {} / {} is {}% - best is {}'.format(epoch, args.num_of_epochs,
                                                                          100 * correct_val / total_val, best_accuracy))


    # save model and ambiguity_thresholds
    torch.save(model.state_dict(), os.path.join(saving_path, 'model_parameters.pth'))
    save_anneal_params(model, args)
    print('Finished Training')


if __name__ == '__main__':
    main()
