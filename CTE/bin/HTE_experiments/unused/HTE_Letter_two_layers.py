import numpy as np
import argparse
import os
import time
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.HTE_two_layers import HTE
import torch
import torch.nn as nn
from Sandboxes.Farjon.CTE.utils.datasets import Letter_dataset
from torch import optim
import matplotlib.pyplot as plt
from itertools import count
from Sandboxes.Farjon.CTE.utils.help_funcs import save_anneal_params, load_anneal_params
from Sandboxes.Farjon.CTE.bin.HTE_experiments.training_functions import train_loop
from Sandboxes.Farjon.CTE.utils.datasets.create_letters_dataset import main as create_letters_dataset

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
    experiment_name = 'HTE-Letter-Recognition_2_layers'
    experiment_number = '1'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'HTE_pytorch', experiment_name,
                                        experiment_number)

    # optimization Parameters
    args.word_calc_learning_rate = 0.001
    args.voting_table_learning_rate = 0.1

    args.LR_decay = 0.999
    args.num_of_epochs = 50
    args.batch_size = 200
    args.optimizer = 'ADAM'
    args.loss = 'categorical_crossentropy'

    args.datadir = os.path.join(GetEnvVar('DatasetsPath'), 'HTE_Omri_Shira', 'LETTER')

    args.datapath = os.path.join(args.datadir, 'split_data')
    train_path, val_path, test_path = create_letters_dataset(args)

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 0}

    # create train,val,test data_loader
    training_set = Letter_dataset.Letters(train_path)
    train_loader = torch.utils.data.DataLoader(training_set, **params)

    train_mean = train_loader.dataset.mean
    train_std = train_loader.dataset.std

    validation_set = Letter_dataset.Letters(val_path, train_mean, train_std)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params)

    testing_set = Letter_dataset.Letters(test_path, train_mean, train_std)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)

    # Letter recognition dataset has 16 features
    D_in = 16
    D_out_1 = 20
    D_out_2 = 26

    args.input_size = [args.batch_size, D_in]
    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch sizes
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern1 = {'K': 10, 'M': 5, 'num_of_features': D_in}
    args.ST1 = {'Num_of_active_words': 2**args.Fern1['K'], 'D_out': D_out_1}

    args.Fern2 = {'K': 10, 'M': 5, 'num_of_features': D_out_1}
    args.ST2 = {'Num_of_active_words': 2**args.Fern1['K'], 'D_out': D_out_2}

    args.prune_type = 1
    args.number_of_layer = 2

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    model = HTE(args, args.input_size, device)

    voting_table_LR_params_list = ['voting_table1.weights', 'voting_table2.weights', 'voting_table1.bias',
                                   'voting_table2.bias']
    voting_table_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
    word_calc_params = list(map(lambda x: x[1], list(
        filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))
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
        save_anneal_params(model.word_calc1, path_to_AT, path_to_anneal_params)
        # save second layer
        path_to_AT = paths_to_save[2]
        path_to_anneal_params = paths_to_save[3]
        save_anneal_params(model.word_calc2, path_to_AT, path_to_anneal_params)

    def load_model_anneal_params(model, paths_to_save):
        # load first layer
        path_to_AT = paths_to_save[0]
        path_to_anneal_params = paths_to_save[1]
        # AT - ambiguity threshold, AP - anneal params
        AT, AP = load_anneal_params(path_to_AT, path_to_anneal_params)
        model.word_calc1.ambiguity_thresholds = AT
        model.word_calc1.anneal_state_params = AP
        # load second layer
        path_to_AT = paths_to_save[2]
        path_to_anneal_params = paths_to_save[3]
        # AT - ambiguity threshold, AP - anneal params
        AT, AP = load_anneal_params(path_to_AT, path_to_anneal_params)
        model.word_calc2.ambiguity_thresholds = AT
        model.word_calc2.anneal_state_params = AP
        return model

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    final_model = train_loop(args, train_loader, validation_loader, model, optimizer, criterion, device, saving_path, save_model_anneal_params)
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
    # best_accuracy = 0
    # model.to(device)
    #
    # # plotting variables
    # th_values = []
    # AT_values = []
    # x_values = []
    # alpha_1 = []
    # alpha_2 = []
    # accuracy_graph = []
    # loss_for_epoch =[]
    # index = count()
    # end = time.time()
    # for epoch in range(args.num_of_epochs):
    #     running_loss_graph = 0.0
    #     for inputs, labels in train_loader:
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         running_loss_graph += float(loss.item())
    #         loss.backward()
    #         optimizer.step()
    #         model.on_batch_ends(device)
    #         print('batch %d loss: %.3f' %
    #               (epoch + 1, loss))
    #         print('batch time is %s' % (time.time() - end))
    #         end = time.time()
    #         optimizer.defaults['lr'] = optimizer.defaults['lr'] * args.LR_decay
    #
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs_val, labels_val in validation_loader:
    #             inputs_val = inputs_val.to(device)
    #             labels_val = labels_val.to(device)
    #             outputs_val = model(inputs_val)
    #             _, predicted = torch.max(outputs_val.data, 1)
    #             total += labels_val.size(0)
    #             correct += (predicted == labels_val).sum().item()
    #
    #             # print statistics
    #         print('Accuracy of the network on the %d validation examples: %.2f %%' % (validation_loader.dataset.examples.shape[0],
    #                 100 * correct / total))
    #         if best_accuracy < (correct / total):
    #             torch.save(model.state_dict(), os.path.join(saving_path, 'best_model_parameters.pth'))
    #     if args.debug and args.visu_parameters:
    #         # plot alphas, th, ambiguity_th
    #         accuracy_graph.append((correct / total))
    #         loss_for_epoch.append((running_loss_graph / labels_val.size(0)))
    #         alpha_values = model.word_calc1.alpha[0, 0].detach().cpu().numpy()
    #         alpha_1.append(alpha_values[0])
    #         alpha_2.append(alpha_values[1])
    #         th_values.append(model.word_calc1.th[0][0].detach().cpu().numpy())
    #         AT_values.append(model.word_calc1.ambiguity_thresholds[0][0][0].detach().cpu().numpy())
    #         x_values.append(next(index))
    #         # plt.plot(x_values, th_values, label='BF thresh')
    #         # plt.plot(x_values, AT_values, label='Ambi thresh')
    #         # plt.plot(x_values, alpha_1, label='BF weight 1')
    #         # plt.plot(x_values, alpha_2, label='BF weight 2')
    #         plt.plot(x_values, accuracy_graph, label='accuracy')
    #         plt.plot(x_values, loss_for_epoch, label='loss')
    #         plt.title('Performance at epoch {} is {}%'.format(epoch, 100 * correct / total))
    #         plt.xlim(0, args.num_of_epochs)
    #         plt.ylim(0, 1)
    #         plt.legend()
    #         plt.draw()
    #         if epoch == args.num_of_epochs - 1:
    #             plt.savefig(os.path.join(saving_path, 'progress plot.png'))
    #         plt.pause(.3)
    #         plt.cla()
    #
    #     # on_batch_ends - callback
    #
    # with torch.no_grad():
    #     for inputs_test, labels_test in test_loader:
    #         inputs_test = inputs_test.to(device)
    #         labels_test = labels_test.to(device)
    #         outputs_test = model(inputs_test)
    #         _, predicted = torch.max(outputs_test.data, 1)
    #         total += labels_test.size(0)
    #         correct += (predicted == labels_test).sum().item()
    #
    #         # print statistics
    #     print('Accuracy of the network on the %d test examples: %.2f %%' % (
    #     test_loader.dataset.examples.shape[0],
    #     100 * correct / total))
    #
    # #save model and ambiguity_thresholds
    # torch.save(model.state_dict(), os.path.join(saving_path, 'final_model_parameters.pth'))
    # # save_anneal_params(model, args)
    #
    # if args.debug and D_in == 2:
    #     color_map = ['red', 'blue']
    #     colors = []
    #     for i in range(len(y_train[:1000])):
    #         colors.append(color_map[int(y_train[i])])
    #     plt.scatter(x_train[:1000, 0], x_train[:1000, 1], s=20, c=colors)
    #     plt.title("Experiment {} with seperation factor {}".format('A', separation_factor))
    #     plt.xlabel("x1")
    #     plt.ylabel("x2")
    #     if args.draw_line:
    #         for i in range (args.Fern1['K']):
    #             w_values = model.word_calc1.alpha[0, i]
    #             w_with_tempature = torch.mul(w_values, model.word_calc1.anneal_state_params['tempature'])
    #             softmax_w = torch.nn.functional.softmax(w_with_tempature)
    #             w1 = softmax_w[0].detach().cpu().numpy()
    #             w2 = softmax_w[1].detach().cpu().numpy() + 10e-5
    #             th = model.word_calc1.th[0][i].detach().cpu().numpy()
    #             x1 = np.linspace(-4, 4, 100)
    #             x2 = (-1 * w1 * x1 + th)/w2
    #             plt.plot(x1, x2, '-g')
    #             plt.ylim(-4, 4)
    #             plt.xlabel('x1', color='#1C2833')
    #             plt.ylabel('x2', color='#1C2833')
    #
    #         plt.savefig(os.path.join(saving_path, 'separation_fig.png'))
    #
    # print('Finished Training')


if __name__ == '__main__':
    main()

