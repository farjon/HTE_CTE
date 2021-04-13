import numpy as np
import argparse
import os
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.CTE_two_layers_inter_loss import CTE as CTE_two_layers
from Sandboxes.Farjon.CTE.models.CTE_single_layer import CTE as CTE_one_layer
from Sandboxes.Farjon.CTE.utils.datasets import get_train_valid_loader, get_test_loader
import torch
import torch.optim as optim
import torch.nn as nn
from Sandboxes.Farjon.CTE.utils.help_funcs import save_anneal_params, load_anneal_params
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
    experiment_name = 'MNIST'
    experiment_number = '2'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name,
                                        experiment_number)

    # init_weights_base_dir = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', 'init ferns and S2D')

    args.layer1_fern_path = None
    args.layer1_S2D_path = None

    args.layer2_fern_path = None
    args.layer2_S2D_path = None

    # optimization Parameters
    args.word_calc_learning_rate = 0.001
    args.voting_table_learning_rate = 0.1

    args.LR_decay = 0.99
    args.num_of_epochs = [1,1]
    args.batch_size = 200
    args.loss = 'categorical_crossentropy'

    datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')

    train_loader, val_loader = get_train_valid_loader(datapath, args.batch_size)
    test_loader = get_test_loader(datapath, args.batch_size)

    H = 28
    W = 28
    C = 1
    args.input_size = (H, W, C)

    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern1 = {'K': 10, 'M': 10, 'L': 15}
    args.ST1 = {'Num_of_active_words': 2**(args.Fern1['K']-5), 'D_out': 10}
    args.AvgPool1 = {'kernel_size' : 3}
    args.AvgPool1_1 = {'kernel_size' : 14}

    args.Fern2 = {'K': 10, 'M': 10, 'L': 5}
    args.ST2 = {'Num_of_active_words': 2**(args.Fern2['K']-5), 'D_out': 10}
    args.AvgPool2 = {'kernel_size': 3}

    args.prune_type = 1

    args.number_of_layers = 1

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    for round in range(len(args.num_of_epochs)):
        num_of_epochs = args.num_of_epochs[round]
        saving_path = os.path.join(args.save_path, str(round))
        if round == 0:
            # first round - create one-layer model
            model = CTE_one_layer(args, [H, W, args.batch_size], device)
            voting_table_LR_params_list = ['voting_table1.weights','voting_table1.bias']
            voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
            word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))
            optimizer = optim.Adam([{'params': word_calc_params},
                                    {'params': voting_table_params, 'lr': args.voting_table_learning_rate}], lr=args.word_calc_learning_rate)

            # create saving paths for best and final models
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            args.layer1_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_1.npy')
            args.layer1_word_calc_anneal_params = os.path.join(saving_path, 'first_layer_anneal_params.p')
            update_layers = [1]
        elif round == 1:
            args.LR_decay = 0.99
            load_path = os.path.join(args.save_path, str(round-1))
            model = CTE_two_layers(args, [H, W, args.batch_size], device)
            # load model parameters and ambiguity threshold parameters
            model.load_state_dict(torch.load(os.path.join(load_path, 'final_model_parameters.pth')), strict=False)
            AT, AP = load_anneal_params(args.layer1_word_calc_ambiguity_thresholds, args.layer1_word_calc_anneal_params)
            model.word_calc1.ambiguity_thresholds = AT
            model.word_calc1.anneal_state_params = AP
            # disable learning of the first layer's parameters
            zero_LR_weights_list = ['word_calc1.dx1', 'word_calc1.dx2', 'word_calc1.dy1', 'word_calc1.dy2', 'word_calc1.th', 'voting_table1.weights', 'voting_table1.bias']
            voting_table_LR_params_list = ['voting_table2.weights', 'voting_table2.bias']
            voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
            word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))
            optimizer = optim.Adam([{'params': word_calc_params},
                                    {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                                   lr=args.word_calc_learning_rate)

            for param in model.named_parameters():
                if param[0] in zero_LR_weights_list:
                    param[1].requires_grad = False

            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            args.layer2_word_calc_ambiguity_thresholds = os.path.join(args.save_path, 'ambiguity_thresholds_layer_2.npy')
            args.layer2_word_calc_anneal_params = os.path.join(args.save_path, 'sec_layer_anneal_params.p')
            update_layers = [0,1]
        elif round == 2:
            load_path = os.path.join(args.save_path, str(round - 1))
            model = CTE_two_layers(args, [H, W, args.batch_size], device)
            model.load_state_dict(torch.load(os.path.join(load_path, 'best_model_parameters.pth')), strict=False)
            voting_table_LR_params_list = ['voting_table1.weights', 'voting_table2.weights', 'voting_table1.bias', 'voting_table2.bias']
            voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
            word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))
            optimizer = optim.Adam([{'params': word_calc_params},
                                    {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                                   lr=args.word_calc_learning_rate)

            args.layer1_word_calc_ambiguity_thresholds = os.path.join(args.save_path, 'ambiguity_thresholds_layer_1.npy')
            args.layer1_word_calc_anneal_params = os.path.join(args.save_path, 'first_layer_anneal_params.p')
        best_accuracy = 0
        # plotting variables
        accuracy_graph = []
        loss_for_epoch = []
        x_values = []
        index = count()

        for epoch in range(num_of_epochs):
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
                if round == 0:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                elif round == 1:
                    outputs, inter_outputs = model(inputs)
                    loss = criterion(outputs, labels)
                else:
                    outputs, inter_outputs = model(inputs)
                    loss1 = criterion(inter_outputs, labels)
                    loss2 = criterion(outputs, labels)
                    loss = loss1 + loss2
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
                model.on_batch_ends(update_layers)
                #lowering learning rate of W,V after each batch
                optimizer.defaults['lr'] = optimizer.defaults['lr'] * args.LR_decay

            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    if round == 0:
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        outputs, _ = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                if args.debug:
                    accuracy_graph.append((correct_val / total_val))
                    loss_for_epoch.append((running_loss_graph / 12000))
                    x_values.append(next(index))
                    plt.plot(x_values, accuracy_graph, label='accuracy')
                    plt.plot(x_values, loss_for_epoch, label='loss')
                    plt.legend()
                    plt.xlim(0, num_of_epochs + 4)
                    plt.ylim(0, 1)
                    plt.title('Experiment 1 Accuracy over epochs')
                    plt.draw()
                    if epoch == num_of_epochs - 1:
                        plt.savefig(os.path.join(saving_path, 'progress plot.png'))
                    plt.pause(.3)
                    plt.cla()
            print('Accuracy of the network on the 12000 validation images: %.2f %%' % (
                    100 * correct_val / total_val))
            if best_accuracy <= (correct_val / total_val):
                torch.save(model.state_dict(), os.path.join(saving_path, 'best_model_parameters.pth'))
                best_accuracy = correct_val/total_val
                if round == 0:
                    save_anneal_params(model.word_calc1, args.layer1_word_calc_ambiguity_thresholds,
                                       args.layer1_word_calc_anneal_params)
                elif round == 1:
                    save_anneal_params(model.word_calc2, args.layer2_word_calc_ambiguity_thresholds,
                                       args.layer2_word_calc_anneal_params)

        if round == 0:
            save_anneal_params(model.word_calc1, args.layer1_word_calc_ambiguity_thresholds,
                               args.layer1_word_calc_anneal_params)
        elif round == 1:
            save_anneal_params(model.word_calc2, args.layer2_word_calc_ambiguity_thresholds,
                               args.layer2_word_calc_anneal_params)

        torch.save(model.state_dict(), os.path.join(saving_path, 'final_model_parameters.pth'))

    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, inter_outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    print('Accuracy test set is {}'.format(100 * correct_val / total_val))


    # save model and ambiguity_thresholds
    torch.save(model.state_dict(), os.path.join(saving_path, 'final_model_parameters.pth'))
    # save_anneal_params(model, args)
    print('Finished Training')


if __name__ == '__main__':
    main()
