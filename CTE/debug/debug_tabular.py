import numpy as np
import argparse
import os
import time
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.HTE_tabluar_single_layer import HTE
import torch
import torch.nn as nn
from Sandboxes.Farjon.CTE.utils.visualize_function import visualize_network_parameters as visualize
from Sandboxes.Farjon.CTE.utils.help_funcs import to_numpy, save_anneal_params, load_anneal_params
from Sandboxes.Farjon.CTE.utils.CTEOptimizer import CTEOptimizer
import matplotlib.pyplot as plt


def create_syn_data(n, D_in, separation_factor, ratio = 0.5, selected_feature = None):
    '''
    this function creates synthetic data generated from Gaussian distribution
    the data will be generated from 2 different distributions for classification tasks
    :param n: the number of examples to generate in total
    :param D_in: the number of features for each example
    :param ratio: ration between the two classes. defualt is 0.5
    :return: x, y - examples and labels. size(x) = n*D_in and size(y) = n
    '''
    # dummy data - data will be generated from noise. a single feature will fully separate between the classes
    x = np.random.normal(0, 1, [n, D_in])
    shuffle_inds = np.random.permutation(n)
    class_1_inds = shuffle_inds[:int(np.floor(n*ratio))]
    class_2_inds = shuffle_inds[int(np.floor(n*ratio)):]
    if selected_feature is None:
        selected_feature = np.random.randint(0,D_in,1)
    x[class_1_inds, selected_feature] = x[class_1_inds, selected_feature] + separation_factor
    x[class_2_inds, selected_feature] = x[class_2_inds, selected_feature]
    y = np.zeros(n, dtype=np.int_)
    y[class_1_inds] = 1
    y[class_2_inds] = 0
    return x, y, selected_feature


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(15)
    torch.manual_seed(0)


    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="HTE model")
    args = parser.parse_args()

    # create opts struct for all optimization parameters
    opts_parser = argparse.ArgumentParser(description="optimization parameters")
    opts = opts_parser.parse_args()

    #debug_mode - also used to visualize and follow specific parameters. See ../CTE/utils/visualize_function.py
    args.debug = False

    # path to save models
    experiment_name = 'hierarchical_table_ensembles'
    experiment_number = 'A'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name,
                                        experiment_number)


    # optimization Parameters
    opts.momentum = 1
    opts.alpha = 0.9
    opts.learning_rate = 0.001
    opts.weight_decay = 5e-4

    args.word_calc_learning_rate = 5e-2
    args.word_calc_weight_decay = 0.1

    args.voting_table_learning_rate = 0.1
    args.voting_table_weight_decay = 0.1

    args.LR_decay = 0.99
    args.num_of_epochs = 200
    args.batch_size = 100000
    args.optimizer = 'CTEOptimizer' #SGD / ADAM / RMSPROP / ADAMW / CTEOptimizer
    args.loss = 'categorical_crossentropy'

    # datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')

    classes = ('0', '1')

    N = 100000
    D_in = 2
    N_test = 100000
    train_test_ratio = 0.5
    separation_factor = 4
    x, y, selected_feature = create_syn_data(N, D_in, separation_factor, train_test_ratio)
    x_test, y_test, selected_feature = create_syn_data(N_test, D_in, separation_factor, train_test_ratio, selected_feature)

    if args.debug:
        color_map = ['red', 'blue']
        colors = []
        for i in range(len(y)):
            colors.append(color_map[y[i]])
        plt.scatter(x[:, 0], x[:, 1], s=20, c=colors)
        plt.title(f"Example of a mixture of 2 distributions")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    args.input_size = [N, D_in]

    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern1 = {'K': 1, 'M': 1, 'num_of_features': D_in}
    args.ST1 = {'Num_of_active_words': 1, 'D_out': 2}

    args.prune_type = 1
    args.number_of_layer = 1

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    end = time.time()
    model = HTE(args, args.input_size, device)

    voting_table_LR_params_list = ['voting_table1.weights', 'voting_table1.bias']
    voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
    word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))

    optimizer = CTEOptimizer([{'params': word_calc_params},
                               {'params': voting_table_params, 'lr': args.voting_table_learning_rate,
                                'alpha': opts.alpha, 'weight_decay': opts.weight_decay,
                                'momentum': opts.momentum, 'RMS_support':False, 'batch_size':args.batch_size}
                               ], lr=args.word_calc_learning_rate, alpha=opts.alpha, weight_decay=opts.weight_decay, momentum=opts.momentum, RMS_support=True, batch_size=args.batch_size)

    saving_path = os.path.join(args.save_path)
    args.layer1_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_1.npy')
    args.layer1_word_calc_anneal_params = os.path.join(saving_path, 'first_layer_anneal_params.p')

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    best_accuracy = 0
    model.to(device)
    for epoch in range(args.num_of_epochs):
        running_loss = 0.0
        # print("Current learning rate is: {}".format(optimizer.param_groups[0]['lr']))

        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.from_numpy(x).to(device)
        labels = torch.from_numpy(y).to(device)
        labels = labels.long()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            inputs_test = torch.from_numpy(x_test).to(device)
            labels_test = torch.from_numpy(y_test).to(device)
            labels_test = labels_test.long()
            outputs_test = model(inputs_test)
            _, predicted = torch.max(outputs_test.data, 1)
            total_val = labels.size(0)
            correct_val = (predicted == labels_test).sum().item()
            # print statistics
            print('Accuracy of the network on the %d validation examples: %.2f %%' % (N,
                    100 * correct_val / total_val))
            if best_accuracy < (correct_val / total_val):
                torch.save(model.state_dict(), os.path.join(saving_path, 'best_model_parameters.pth'))
        print('batch time is %s' %(time.time() - end))
        end = time.time()

        print('epoch %d loss: %.3f' %
              (epoch + 1, loss))
        # on_batch_ends - callback
        model.on_batch_ends(device)

    #save model and ambiguity_thresholds
    torch.save(model.state_dict(), os.path.join(saving_path, 'final_model_parameters.pth'))
    # save_anneal_params(model, args)


    print('Finished Training')


if __name__ == '__main__':
    main()

