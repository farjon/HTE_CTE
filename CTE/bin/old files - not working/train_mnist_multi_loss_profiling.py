import numpy as np
import argparse
import os
import time
import sys
sys.path.append('C:\\Users\\owner\\Documents\\pythonroot_2018\\')
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.CTE_two_layers_inter_loss import CTE
from Sandboxes.Farjon.CTE.utils.datasets import get_train_valid_loader, get_test_loader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.cuda.nvtx as nvtx
from Sandboxes.Farjon.CTE.utils.visualize_function import visualize_network_parameters as visualize
from Sandboxes.Farjon.CTE.utils.help_funcs import to_numpy, save_anneal_params, load_anneal_params
from Sandboxes.Farjon.CTE.utils.CTEOptimizer import CTEOptimizer
# from pytorch_memlab import profile
# from torch.utils.tensorboard import SummaryWriter
import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(10)
    torch.manual_seed(0)


    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="CTE model")
    args = parser.parse_args()

    # create opts struct for all optimization parameters
    opts_parser = argparse.ArgumentParser(description="optimization parameters")
    opts = opts_parser.parse_args()

    #debug_mode - also used to visualize and follow specific parameters. See ../CTE/utils/visualize_function.py
    args.debug = True

    # path to save models
    experiment_name = 'mnist_multiple_losses'
    experiment_number = '0000'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    args.save_graph_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name,
                                        experiment_number)

    init_weights_base_dir = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', 'init ferns and S2D')

    args.layer1_fern_path = os.path.join(init_weights_base_dir, 'Fern_1')
    args.layer1_S2D_path = os.path.join(init_weights_base_dir, 'S2D_1')

    args.layer2_fern_path = os.path.join(init_weights_base_dir, 'Fern_2')
    args.layer2_S2D_path = os.path.join(init_weights_base_dir, 'S2D_2')

    args.transform = True

    # optimization Parameters
    opts.momentum = 1
    opts.alpha = 0.9
    opts.learning_rate = 1
    opts.weight_decay = 5e-5

    args.word_calc_learning_rate = 5e-3
    args.word_calc_weight_decay = 0.1

    args.voting_table_learning_rate = 0.1
    args.voting_table_weight_decay = 0.1

    args.LR_decay = 0.99
    args.num_of_epochs = [1,1,1]
    args.batch_size = 1000
    args.optimizer = 'CTEOptimizer' #SGD / ADAM / RMSPROP / ADAMW / CTEOptimizer
    args.loss = 'categorical_crossentropy'

    # make sure torch is the minimum required version
    # torch_version.check_torch_version()

    datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')

    train_loader, val_loader = get_train_valid_loader(datapath, args.batch_size)

    test_loader = get_test_loader(datapath, args.batch_size)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
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
    args.Fern1 = {'K': 10, 'M': 10, 'L': 13}
    args.ST1 = {'Num_of_active_words': 16, 'D_out': 10}
    args.AvgPool1 = {'kernel_size' : 7}
    args.AvgPool1_1 = {'kernel_size' : 16}

    args.Fern2 = {'K': 10, 'M': 10, 'L': 7}
    args.ST2 = {'Num_of_active_words': 16, 'D_out': 10}
    args.AvgPool2 = {'kernel_size': 4}

    args.prune_type = 1

    args.number_of_layer = 2

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss(reduction='sum')

    end = time.time()
    for index in range(len(args.num_of_epochs)):
    # for index in range(1,3):
        current_num_of_epochs = args.num_of_epochs[index]

        if index == 0:
            inter_loss_w, final_loss_w = 1, 0
            model = CTE(args, [H, W, args.batch_size])
            model.to(device)

            voting_table_LR_params_list = ['voting_table1.weights', 'voting_table2.weights', 'voting_table1.bias', 'voting_table2.bias']
            voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
            word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))

            optimizer = CTEOptimizer([{'params': word_calc_params},
                                       {'params': voting_table_params, 'lr': args.voting_table_learning_rate,
                                        'alpha': opts.alpha, 'weight_decay': opts.weight_decay,
                                        'momentum': opts.momentum, 'RMS_support':False, 'batch_size':args.batch_size}
                                       ], lr=args.word_calc_learning_rate, alpha=opts.alpha, weight_decay=opts.weight_decay, momentum=opts.momentum, RMS_support=True, batch_size=args.batch_size)

            # optimizer_fern = optim.RMSprop(word_calc_params, lr=args.word_calc_learning_rate, weight_decay = args.word_calc_weight_decay)
            # optimizer_VT = optim.SGD(voting_table_params, lr=args.voting_table_learning_rate, weight_decay = args.voting_table_weight_decay)
            #
            # scheduler_fern = torch.optim.lr_scheduler.ExponentialLR(optimizer_fern, gamma=args.LR_decay, last_epoch=-1)
            # scheduler_VT = torch.optim.lr_scheduler.ExponentialLR(optimizer_VT, gamma=args.LR_decay, last_epoch=-1)

            # if args.optimizer == 'SGD':
            #     optimizer = optim.SGD([{'params': word_calc_params},
            #                            {'params': voting_table_params, 'lr': args.voting_table_learning_rate, 'weight_decay' : args.voting_table_weight_decay}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'RMSPROP':
            #     optimizer = optim.RMSprop([{'params': word_calc_params},
            #                            {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'ADAM':
            #     optimizer = optim.Adam([{'params': word_calc_params},
            #                                {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'ADAMW':
            #     optimizer = optim.AdamW([{'params': word_calc_params},
            #                                {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                               lr=args.word_calc_learning_rate)
            #

            model_save_dir = 'first stage'

        elif index == 1:
            # model_save_dir = 'first stage'
            # saving_path = os.path.join(args.save_path, model_save_dir)
            # args.layer1_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_1.npy')
            # args.layer2_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_2.npy')
            # args.layer1_word_calc_anneal_params = os.path.join(saving_path, 'first_layer_anneal_params.p')
            # args.layer2_word_calc_anneal_params = os.path.join(saving_path, 'sec_layer_anneal_params.p')

            inter_loss_w, final_loss_w = 0, 1
            model = CTE(args, [H, W, args.batch_size])
            model.to(device)
            #load weights
            model.load_state_dict(torch.load(os.path.join(saving_path, 'model_parameters.pth')))
            #load ambiguity thresholds
            model = load_anneal_params(model, args)

            freeze_weights_params_list = ['voting_table1.weights', 'word_calc1.dx1', 'word_calc1.dx2', 'word_calc1.dy1', 'word_calc1.dy2', 'word_calc1.th', 'voting_table1.bias']
            different_LR_params_list = ['voting_table2.weights', 'voting_table2.bias']
            freeze_weights_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in freeze_weights_params_list, model.named_parameters()))))
            voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in different_LR_params_list, model.named_parameters()))))
            temp_param_list = different_LR_params_list + freeze_weights_params_list
            word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in temp_param_list, model.named_parameters()))))


            optimizer = CTEOptimizer([{'params': word_calc_params},
                                      {'params': freeze_weights_params, 'lr': 0,
                                       'alpha': opts.alpha, 'weight_decay': opts.weight_decay,
                                       'momentum': opts.momentum, 'RMS_support':False, 'batch_size':args.batch_size},
                                      {'params': voting_table_params, 'lr': args.voting_table_learning_rate,
                                       'alpha': opts.alpha, 'weight_decay': opts.weight_decay,
                                       'momentum': opts.momentum, 'RMS_support': False, 'batch_size':args.batch_size}
                                       ],
                                     lr=args.word_calc_learning_rate, alpha=opts.alpha, weight_decay=opts.weight_decay, momentum=opts.momentum, RMS_support=True, batch_size=args.batch_size)

            # optimizer_fern = optim.RMSprop(word_calc_params, lr=args.word_calc_learning_rate,
            #                                weight_decay=args.word_calc_weight_decay)
            # optimizer_VT = optim.SGD(voting_table_params, lr=args.voting_table_learning_rate,
            #                          weight_decay=args.voting_table_weight_decay)
            #
            # scheduler_fern = torch.optim.lr_scheduler.ExponentialLR(optimizer_fern, gamma=args.LR_decay, last_epoch=-1)
            # scheduler_VT = torch.optim.lr_scheduler.ExponentialLR(optimizer_VT, gamma=args.LR_decay, last_epoch=-1)


            # if args.optimizer == 'SGD':
            #     optimizer = optim.SGD([{'params': word_calc_params},
            #                            {'params': voting_table_params, 'lr': args.voting_table_learning_rate},
            #                                {'params' : freeze_weights_params, 'lr': 0}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'RMSPROP':
            #     optimizer = optim.RMSprop([{'params': word_calc_params},
            #                                {'params': voting_table_params, 'lr': args.voting_table_learning_rate},
            #                                {'params' : freeze_weights_params, 'lr': 0}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'ADAM':
            #     optimizer = optim.Adam([{'params': word_calc_params},
            #                                {'params': voting_table_params, 'lr': args.voting_table_learning_rate},
            #                                {'params' : freeze_weights_params, 'lr': 0}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'ADAMW':
            #     optimizer = optim.AdamW([{'params': word_calc_params},
            #                             {'params': voting_table_params, 'lr': args.voting_table_learning_rate},
            #                             {'params': freeze_weights_params, 'lr': 0}],
            #                            lr=args.word_calc_learning_rate)
            #
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay, last_epoch=-1)
            model_save_dir = 'second stage'
        else:
            inter_loss_w, final_loss_w = 0.5, 0.5
            model = CTE(args, [H, W, args.batch_size])
            model.load_state_dict(torch.load(os.path.join(saving_path, 'model_parameters.pth')))
            model = load_anneal_params(model, args)
            model.to(device)

            different_LR_params_list = ['voting_table1.weights', 'voting_table2.weights', 'voting_table1.bias', 'voting_table2.bias']
            voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in different_LR_params_list, model.named_parameters()))))
            word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in different_LR_params_list, model.named_parameters()))))

            optimizer = CTEOptimizer([{'params': word_calc_params},
                                      {'params': voting_table_params, 'lr': args.voting_table_learning_rate,
                                       'alpha': opts.alpha, 'weight_decay': opts.weight_decay,
                                       'momentum': opts.momentum, 'RMS_support': False, 'batch_size':args.batch_size}
                                       ],
                                     lr=args.word_calc_learning_rate, alpha=opts.alpha, weight_decay=opts.weight_decay, momentum=opts.momentum, RMS_support=True, batch_size=args.batch_size)
            # optimizer = optim.RMSprop([{'params': word_calc_params}, {'params': voting_table_params, 'lr': '0.1'}], lr=1, momentum=0.9)
            # optimizer_fern = optim.RMSprop(word_calc_params, lr=args.word_calc_learning_rate,
            #                                weight_decay=args.word_calc_weight_decay)
            # optimizer_VT = optim.SGD(voting_table_params, lr=args.voting_table_learning_rate,
            #                          weight_decay=args.voting_table_weight_decay)
            #
            # scheduler_fern = torch.optim.lr_scheduler.ExponentialLR(optimizer_fern, gamma=args.LR_decay, last_epoch=-1)
            # scheduler_VT = torch.optim.lr_scheduler.ExponentialLR(optimizer_VT, gamma=args.LR_decay, last_epoch=-1)

            # if args.optimizer == 'SGD':
            #     optimizer = optim.SGD([{'params': word_calc_params},
            #                            {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'RMSPROP':
            #     optimizer = optim.RMSprop([{'params': word_calc_params},
            #                            {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                               lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'ADAM':
            #     optimizer = optim.Adam([{'params': word_calc_params},
            #                             {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                            lr=args.word_calc_learning_rate)
            # elif args.optimizer == 'ADAMW':
            #     optimizer = optim.AdamW([{'params': word_calc_params},
            #                             {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
            #                            lr=args.word_calc_learning_rate)
            #
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay, last_epoch=-1)

            model_save_dir = 'final stage'

        saving_path = os.path.join(args.save_path, model_save_dir)
        args.layer1_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_1.npy')
        args.layer2_word_calc_ambiguity_thresholds = os.path.join(saving_path, 'ambiguity_thresholds_layer_2.npy')
        args.layer1_word_calc_anneal_params = os.path.join(saving_path, 'first_layer_anneal_params.p')
        args.layer2_word_calc_anneal_params = os.path.join(saving_path, 'sec_layer_anneal_params.p')

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        index_for_graph = 0

        if args.debug:
            fig, ax, lines = None, None, None
            loss = torch.tensor([0])
        with torch.autograd.profiler.emit_nvtx():

            for epoch in range(current_num_of_epochs):
                running_loss = 0.0
                # print("Current learning rate is: {}".format(optimizer.param_groups[0]['lr']))
                for i, data in enumerate(train_loader, 0):
                    if args.debug and (i % 10 == 9 or i == 0):
                        vis_args = {'stage': index, 'batch': i, 'epoch': epoch, 'graph_path': args.save_graph_path, 'index': index_for_graph}
                        current_rho = model .word_calc1.anneal_state_params['Rho']
                        word_calc1_bit1_params = [to_numpy(model.word_calc1.dx1[3,3]),
                                                   to_numpy(model.word_calc1.dy1[3,3]),
                                                   to_numpy(model.word_calc1.dx2[3,3]),
                                                   to_numpy(model.word_calc1.dy2[3,3])]
                        word_calc1_bit2_params = [to_numpy(model.word_calc1.dx1[5,7]),
                                                   to_numpy(model.word_calc1.dy1[5,7]),
                                                   to_numpy(model.word_calc1.dx2[5,7]),
                                                   to_numpy(model.word_calc1.dy2[5,7])]
                        patch_cetner = np.round((args.Fern1['L']/2),0)
                        loss_value = loss.data
                        fig, ax, lines = visualize(vis_args, current_rho, word_calc1_bit1_params, word_calc1_bit2_params, patch_cetner, loss_value, fig, ax, lines)
                    index_for_graph = index_for_graph + 1
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = torch.round(inputs)
                    inputs, labels = inputs.to(device), labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    # profiler.start()
                    outputs, inter_outputs = model(inputs)
                    loss1 = criterion(inter_outputs, labels)
                    loss2 = criterion(outputs, labels)
                    loss = inter_loss_w*loss1 + final_loss_w*loss2
                    loss.backward()
                    optimizer.step()
                    # profiler.stop()
                    running_loss += float(loss.item())
                    # print statistics
                    print('batch time is %s' %(time.time() - end))
                    end = time.time()
                    if i % 10 == 9:  # print every 10 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 10))
                        running_loss = 0.0
                    # on_batch_ends - callback
                    model.on_batch_ends()
                # scheduler_fern.step()
                # scheduler_VT.step()

            #save model and ambiguity_thresholds
            torch.save(model.state_dict(), os.path.join(saving_path, 'model_parameters.pth'))
            save_anneal_params(model, args)


        #     if index == 0 or index == 1:
        #         correct_val = 0
        #         total_val = 0
        #         with torch.no_grad():
        #             for data in val_loader:
        #                 inputs, labels = data
        #                 inputs, labels = inputs.to('cuda'), labels.to('cuda')
        #                 outputs, inter_outputs = model(inputs)
        #                 if index == 0:
        #                     _, predicted = torch.max(inter_outputs.data, 1)
        #                 else:
        #                     _, predicted = torch.max(outputs.data, 1)
        #                 total_val += labels.size(0)
        #                 correct_val += (predicted == labels).sum().item()
        #
        #         print('Accuracy of the network on the 12000 validation images: %.2f %%' % (
        #                 100 * correct_val / total_val))
        #
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for data in test_loader:
        #         inputs, labels = data
        #         inputs, labels = inputs.to('cuda'), labels.to('cuda')
        #         outputs, inter_outputs = model(inputs)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        #         100 * correct / total))

        print('Finished Training')


if __name__ == '__main__':
    main()
