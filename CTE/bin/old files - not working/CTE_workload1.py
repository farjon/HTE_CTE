
# a profiling workload based om train_mnist_multi_loss.py, but finishes after a few batches
import numpy as np
import argparse
import os
import time
from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.models.CTE_two_layers_inter_loss import CTE
from Sandboxes.Farjon.CTE.utils.datasets import get_train_valid_loader, get_test_loader
import torch
import torch.optim as optim
import torch.nn as nn
# from pytorch_memlab import profile
# from torch.utils.tensorboard import SummaryWriter

def main():
    torch.cuda.set_device(0)
    np.random.seed(10)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="CTE model")
    args = parser.parse_args()

    # path to save models
    experiment_name = 'mnist_multiple_losses'
    experiment_number = '0'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_name, experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optimization Parameters
    args.transform = False
    args.learning_rate = 1
    args.momentum = 0
    args.weightDecay = 0
    args.LR_decay = 0.95
    args.num_of_epochs = [1,50,50]
    args.batch_size = 1200
    args.optimizer = 'SGD' #SGD / ADAM / RMSPROP
    args.loss = 'categorical_crossentropy'

    # make sure torch is the minimum required version
    # torch_version.check_torch_version()

    datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')

    train_loader, val_loader = get_train_valid_loader(datapath, args.batch_size,  num_workers = 1,  pin_memory = True)



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

    args.number_of_layer = 2

    if args.loss == 'categorical_crossentropy':
        criterion = nn.CrossEntropyLoss()



    end = time.time()
    for index in range(0,1):
        current_num_of_epochs = args.num_of_epochs[index]

        if index == 0:
            inter_loss_w, final_loss_w = 1, 0
            model = CTE(args, [H, W, args.batch_size])
            model_parameters = model.parameters()
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(model_parameters, lr=args.learning_rate, momentum=args.momentum)

            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay, last_epoch=-1)

            model_save_dir = 'first stage'

        elif index == 1:
            inter_loss_w, final_loss_w = 0, 1
            model = CTE(args, [H, W, args.batch_size])
            model.load_state_dict(torch.load(os.path.join(saving_path, 'model_parameters.pth')))
            model_parameters = model.parameters()
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(model_parameters, lr=args.learning_rate, momentum=args.momentum)

            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay, last_epoch=-1)

            model_save_dir = 'second stage'
        else:
            inter_loss_w, final_loss_w = 0.5, 0.5
            model = CTE(args, [H, W, args.batch_size])
            model.load_state_dict(torch.load(os.path.join(saving_path, 'model_parameters.pth')))
            model_parameters = model.parameters()
            if args.optimizer == 'SGD':
                optimizer = optim.SGD(model_parameters, lr=args.learning_rate, momentum=args.momentum)

            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.LR_decay, last_epoch=-1)

            model_save_dir = 'final stage'

        saving_path = os.path.join(args.save_path, model_save_dir)
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        for epoch in range(current_num_of_epochs):
            running_loss = 0.0
            print("Current learning rate is: {}".format(optimizer.param_groups[0]['lr']))
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                outputs, inter_outputs = model(inputs)
                # print(prof)
                loss1 = criterion(inter_outputs, labels)
                loss2 = criterion(outputs, labels)
                loss = inter_loss_w*loss1 + final_loss_w*loss2
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                # print statistics
                print('batch time is %s' %(time.time() - end))
                end = time.time()
                print("i = ", i)
                if i == 2:
                    break
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

                # on_batch_ends  callback
                model.on_batch_ends()

            scheduler.step()


        torch.save(model.state_dict(), os.path.join(saving_path, 'model_parameters.pth'))



    print('Finished Training')


if __name__ == '__main__':
    main()
