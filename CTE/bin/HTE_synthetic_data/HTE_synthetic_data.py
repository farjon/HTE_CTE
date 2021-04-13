import numpy as np
import argparse
import os
import time
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.HTE_tabluar_single_layer import HTE
import torch
import torch.nn as nn
from Sandboxes.Farjon.CTE.utils.help_funcs import to_numpy, save_anneal_params, load_anneal_params
from Sandboxes.Farjon.CTE.utils.create_synthetic_data import create_syn_data, create_pariti_bit_syn_data
from Sandboxes.Farjon.CTE.utils.CTEOptimizer import CTEOptimizer
from torch import optim
import matplotlib.pyplot as plt
from itertools import count

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

    # create opts struct for all optimization parameters
    opts_parser = argparse.ArgumentParser(description="optimization parameters")
    opts = opts_parser.parse_args()

    #debug_mode - also used to visualize and follow specific parameters. See ../CTE/utils/visualize_function.py
    args.debug = True
    args.visu_parameters = False
    args.draw_line = True

    # path to save models
    experiment_name = 'hierarchical_table_ensembles'
    experiment_number = 'C'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name,
                                        experiment_number)

    # optimization Parameters
    opts.momentum = 1
    opts.alpha = 0.9
    opts.learning_rate = 1
    opts.weight_decay = 5e-4

    args.word_calc_learning_rate = 5e-3
    args.word_calc_weight_decay = 0.1

    args.voting_table_learning_rate = 0.1
    args.voting_table_weight_decay = 0.1

    args.LR_decay = 0.99
    args.num_of_epochs = 300
    args.batch_size = 200
    args.optimizer = 'ADAM' #SGD / ADAM / RMSPROP / ADAMW / CTEOptimizer
    args.loss = 'categorical_crossentropy'

    # datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')

    classes = ('0', '1')

    # ------------- Start editing here -------------
    # this parameters are related to the creation of synthetic data
    choose_exp = 'C' # choose between A,B,C for Gaussian data, D for parity-function data

    if choose_exp is 'A' or choose_exp is 'B' or choose_exp is 'C':
        if choose_exp is 'A':
            num_of_dims = 2
        elif choose_exp is 'B':
            num_of_dims = 30
        elif choose_exp is 'C':
            num_of_dims = 2

        N = 10000
        D_in = num_of_dims
        classes_ratio = 0.5
        train_test_split_factor = 0.8
        separation_factor = 2

        mean_x1 = np.zeros(D_in)
        mean_x2 = np.zeros(D_in)
        mean_x2[0] = separation_factor
        if choose_exp is 'C':
            mean_x2[1] = separation_factor
        cov_x1 = np.eye(D_in)
        cov_x2 = np.eye(D_in)
        means = np.array([mean_x1, mean_x2])
        covs = np.array([cov_x1, cov_x2])

        x_train, x_test, y_train, y_test = create_syn_data(N, means, covs, classes_ratio, train_test_split_factor)

    elif choose_exp is 'D':
        N = 10000
        D_in = 2
        train_test_split_factor = 0.8
        N_train = int(np.round(N*train_test_split_factor))
        N_test = N - N_train
        means = np.zeros(D_in)
        covs = np.eye(D_in)

        x_train, y_train = create_pariti_bit_syn_data(N_train, means, covs)
        x_test, y_test = create_pariti_bit_syn_data(N_test, means, covs)


    # ------------- End editing here -------------

    # Normalize data
    # for now - std for each Gaussian is already 1, only move mean
    if choose_exp is not 'D':
        x_train_mean = np.average(x_train,0)
        x_train = x_train - x_train_mean
        x_test = x_test - x_train_mean

    # if args.debug and D_in == 2:
    #     color_map = ['red', 'blue']
    #     colors = []
    #     for i in range(len(y_train[:1000])):
    #         colors.append(color_map[int(y_train[i])])
    #     plt.scatter(x_train[:1000, 0], x_train[:1000, 1], s=20, c=colors)
    #     plt.title("Experiment {} with seperation factor {}".format(choose_exp, separation_factor))
    #     plt.xlabel("x1")
    #     plt.ylabel("x2")
    #     plt.show()

    args.input_size = [args.batch_size, D_in]
    number_of_batches_train = int(x_train.shape[0]/args.batch_size)
    number_of_batches_test = int(x_test.shape[0]/args.batch_size)
    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern1 = {'K': 2, 'M': 1, 'num_of_features': D_in}
    args.ST1 = {'Num_of_active_words': 2**args.Fern1['K'], 'D_out': 2}

    args.prune_type = 1
    args.number_of_layer = 1

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    end = time.time()
    model = HTE(args, args.input_size, device)

    voting_table_LR_params_list = ['voting_table1.weights', 'voting_table1.bias']
    voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
    word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))

    # optimizer = CTEOptimizer([{'params': word_calc_params},
    #                            {'params': voting_table_params, 'lr': args.voting_table_learning_rate,
    #                             'alpha': opts.alpha, 'weight_decay': opts.weight_decay,
    #                             'momentum': opts.momentum, 'RMS_support':False, 'batch_size':args.batch_size}
    #                            ], lr=args.word_calc_learning_rate, alpha=opts.alpha, weight_decay=opts.weight_decay, momentum=opts.momentum, RMS_support=True, batch_size=args.batch_size)

    optimizer = optim.Adam([{'params': word_calc_params},
                           {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                              lr=args.word_calc_learning_rate)

    saving_path = os.path.join(args.save_path)
    args.layer1_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_1.npy')
    args.layer1_word_calc_anneal_params = os.path.join(saving_path, 'first_layer_anneal_params.p')

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    best_accuracy = 0
    model.to(device)

    # plotting variables
    th_values = []
    AT_values = []
    x_values = []
    alpha_1 = []
    alpha_2 = []
    index = count()
    for epoch in range(args.num_of_epochs):
        for i in range(number_of_batches_train):
            local_x = x_train[i * args.batch_size:(i + 1) * args.batch_size]
            local_y = y_train[i * args.batch_size:(i + 1) * args.batch_size]
            running_loss = 0.0
            # print("Current learning rate is: {}".format(optimizer.param_groups[0]['lr']))

            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.from_numpy(local_x).to(device)
            labels = torch.from_numpy(local_y).to(device)
            labels = labels.long()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            model.on_batch_ends(device)

        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(number_of_batches_test):
                local_x = x_test[i * args.batch_size:(i + 1) * args.batch_size]
                local_y = y_test[i * args.batch_size:(i + 1) * args.batch_size]
                inputs_test = torch.from_numpy(local_x).to(device)
                labels_test = torch.from_numpy(local_y).to(device)
                labels_test = labels_test.long()
                outputs_test = model(inputs_test)
                _, predicted = torch.max(outputs_test.data, 1)
                total += labels_test.size(0)
                correct += (predicted == labels_test).sum().item()

                # print statistics
            print('Accuracy of the network on the %d validation examples: %.2f %%' % (N,
                    100 * correct / total))
            if best_accuracy < (correct / total):
                torch.save(model.state_dict(), os.path.join(saving_path, 'best_model_parameters.pth'))
        print('batch time is %s' %(time.time() - end))
        end = time.time()
        if args.debug and args.visu_parameters:
            # plot alphas, th, ambiguity_th
            alpha_values = model.word_calc1.alpha[0, 0].detach().cpu().numpy()
            alpha_1.append(alpha_values[0])
            alpha_2.append(alpha_values[1])
            th_values.append(model.word_calc1.th[0][0].detach().cpu().numpy())
            AT_values.append(model.word_calc1.ambiguity_thresholds[0][0][0].detach().cpu().numpy())
            x_values.append(next(index))
            plt.plot(x_values, th_values, label='BF thresh')
            plt.plot(x_values, AT_values, label='Ambi thresh')
            plt.plot(x_values, alpha_1, label='BF weight 1')
            plt.plot(x_values, alpha_2, label='BF weight 2')
            plt.title('Performance at epoch {} is {}%'.format(epoch, 100 * correct / total))
            plt.xlim(0, args.num_of_epochs)
            plt.ylim(-5, 5)
            plt.legend()
            plt.draw()
            if epoch == args.num_of_epochs - 1:
                plt.savefig(os.path.join(saving_path, 'progress plot.png'))
            plt.pause(.3)
            plt.cla()

        print('epoch %d loss: %.3f' %
              (epoch + 1, loss))
        # on_batch_ends - callback


    #save model and ambiguity_thresholds
    torch.save(model.state_dict(), os.path.join(saving_path, 'final_model_parameters.pth'))
    # save_anneal_params(model, args)

    if args.debug and D_in == 2:
        color_map = ['red', 'blue']
        colors = []
        for i in range(len(y_train[:1000])):
            colors.append(color_map[int(y_train[i])])
        plt.scatter(x_train[:1000, 0], x_train[:1000, 1], s=20, c=colors)
        if choose_exp =='D':
            plt.title("Experiment {}".format(choose_exp))
        else:
            plt.title("Experiment {} with seperation factor {}".format(choose_exp, separation_factor))
        plt.xlabel("x1")
        plt.ylabel("x2")
        if args.draw_line:
            for i in range (args.Fern1['K']):
                w_values = model.word_calc1.alpha[0, i]
                w_with_tempature = torch.mul(w_values, model.word_calc1.anneal_state_params['tempature'])
                softmax_w = torch.nn.functional.softmax(w_with_tempature)
                w1 = softmax_w[0].detach().cpu().numpy()
                w2 = softmax_w[1].detach().cpu().numpy() + 10e-5
                th = model.word_calc1.th[0][i].detach().cpu().numpy()
                x1 = np.linspace(-4, 4, 100)
                x2 = (-1 * w1 * x1 + th)/w2
                plt.plot(x1, x2, '-g')
                plt.ylim(-4, 4)
                plt.xlabel('x1', color='#1C2833')
                plt.ylabel('x2', color='#1C2833')
        plt.show()

    print('Finished Training')


if __name__ == '__main__':
    main()

