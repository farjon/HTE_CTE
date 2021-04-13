import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from GetEnvVar import GetEnvVar
from CTE_four_layers import CTE
import datasets
import torch
import torch_version
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn



def main():
    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="CTE model for cifar 10 dataset")
    args = parser.parse_args()

    # path to save models
    experiment_number = '0'
    args.save_path = os.path.join(GetEnvVar('ModelsPath'), 'Guy', 'CTE_pytorch', experiment_number)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optimization Parameters
    args.transform = False
    args.learning_rate = 1e-2
    args.momentum = 0.9
    args.weightDecay = 0.0005
    args.num_of_epochs = 20
    args.batch_size = 200
    args.optimizer = 'SGD' #SGD / ADAM / RMSPROP
    args.loss = 'categorical_crossentropy'

    # make sure torch is the minimum required version
    torch_version.check_torch_version()

    transform = transforms.Compose([transforms.ToTensor()])
        # [transforms.ToTensor(),
        #  transforms.Normalize((0.5,), (0.5,))])

    datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')



    train_loader, val_loader = datasets.get_train_valid_loader(datapath, args.batch_size)

    test_loader = datasets.get_test_loader(datapath, args.batch_size)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    H = 32
    W = 32
    C = 3

    args.input_size = (H, W, C)

    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    args.Fern1 = {'K': 10, 'M': 10, 'L': 7}
    args.ST1 = {'Num_of_active_words': 16, 'D_out': 32}
    args.AvgPool1 = {'kernel_size' : 3}

    args.Fern2 = {'K': 10, 'M': 10, 'L': 7}
    args.ST2 = {'Num_of_active_words': 16, 'D_out': 32}
    args.AvgPool2 = {'kernel_size': 3}

    args.Fern3 = {'K': 10, 'M': 10, 'L': 7}
    args.ST3 = {'Num_of_active_words': 16, 'D_out': 64}
    args.AvgPool3 = {'kernel_size' : 3}

    args.Fern4 = {'K': 10, 'M': 10, 'L': 7}
    args.ST4 = {'Num_of_active_words': 16, 'D_out': 10}
    args.AvgPool4 = {'kernel_size': 2}

    args.number_of_layer = 4

    model = CTE(args, len(classes))

    criterion = nn.CrossEntropyLoss()
    model_parameters = model.parameters()
    optimizer = optim.SGD(model_parameters, lr=0.9, momentum=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    for epoch in range(args.num_of_epochs):
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
            outputs = model(inputs)
            # print(prof)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(loss.item())
            # print statistics
            running_loss += float(loss.item())

            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            # on_batch_ends  callback
            model.on_batch_ends()

        scheduler.step()
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        #     model(x)
        # print(prof)

        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 validation images: %d %%' % (
                100 * correct_val / total_val))

    # torch.save(model, args.save_path)
    #
    # test_model = torch.load(args.save_path)
    #
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    print('Finished Training')


if __name__ == '__main__':
    torch.cuda.set_device(0)
    np.random.seed(10)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
