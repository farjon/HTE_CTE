import numpy as np
import argparse
import os

from GetEnvVar import GetEnvVar
from Sandboxes.Farjon.CTE.debug.model_debug import CTE
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

def main():
    np.random.seed(10)

    # create args struct for all parameters
    parser = argparse.ArgumentParser(description="CTE model")
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
    args.num_of_epochs = 100
    args.batch_size = 1
    args.optimizer = 'ADAM' #SGD / ADAM / RMSPROP
    args.loss = 'categorical_crossentropy'

    transform = transforms.Compose([transforms.ToTensor()])
        # [transforms.ToTensor(),
        #  transforms.Normalize((0.5,), (0.5,))])

    datapath = os.path.join(GetEnvVar('DatasetsPath'), 'Mnist_pytorch')



    trainset = torchvision.datasets.MNIST(root=datapath, train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=datapath, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

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
    args.Fern1 = {'K': 4, 'M': 1, 'L': 9}
    args.ST1 = {'Num_of_active_words': 16, 'D_out': 10}

    args.number_of_layer = 1

    model = CTE(args, len(classes))

    criterion = nn.CrossEntropyLoss()
    word_calc_grad = True
    voting_table_grad = False
    fully_layer_gard = False

    model.word_calc1.dx1.requires_grad = word_calc_grad
    model.word_calc1.dx2.requires_grad = word_calc_grad
    model.word_calc1.dy1.requires_grad = word_calc_grad
    model.word_calc1.dy2.requires_grad = word_calc_grad
    model.word_calc1.th.requires_grad = word_calc_grad

    model.voting_table1.weights.requires_grad = voting_table_grad

    model.pred.weight.requires_grad = fully_layer_gard
    model.pred.bias.requires_grad = fully_layer_gard

    model_parameters = model.parameters()
    optimizer = optim.SGD(model_parameters, lr=0.0001, momentum=0)

    for epoch in range(args.num_of_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            # zero the parameter gradients
            # writer = SummaryWriter('run/guy')
            # writer.add_graph(model, inputs)
            # writer.close()
            optimizer.zero_grad()
            for j in range(1000):
            # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                print(loss.item())
                # print(model.word_calc1.dx1)
                # print(model.word_calc1.dx2)
                # print(model.word_calc1.dy1)
                # print(model.word_calc1.dy2)

            print(i)
            # print statistics
            running_loss += float(loss.item())

            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

            # on_batch_ends callback
            model.on_batch_ends()

    print('Finished Training')

if __name__ == '__main__':
    torch.cuda.set_device(0)
    np.random.seed(10)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
