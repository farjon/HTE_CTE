import numpy as np
import os
import time
from GetEnvVar import GetEnvVar
import torch
import matplotlib.pyplot as plt
from itertools import count

def train_loop(args, train_loader, model, optimizer, criterion, device, saving_path, save_anneal_func):
    paths_to_save = args.paths_to_save
    best_accuracy = 0
    model.to(device)
    # plotting variables
    x_values = []
    rho_graph = []
    loss_for_epoch = []
    index = count()
    end = time.time()
    number_of_batchs = args.number_of_batches
    rho = 0
    for epoch in range(args.num_of_epochs):
        #running_loss_graph = 0.0
        batch_index = 0
        for inputs, labels in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            #forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
           # running_loss_graph += float(loss.item())
            loss.backward()
            optimizer.step()
            model.on_batch_ends(device)
            #print('epoch %d/%d, batch %d/%d loss: %.3f,  time: %.3f' %
            #      (epoch + 1, args.num_of_epochs, batch_index + 1, args.number_of_batches, loss, time.time() - end))
            end = time.time()
            batch_index+=1
        print('epoch %d/%d, loss: %.3f,  time: %.3f' %
              (epoch + 1, args.num_of_epochs, loss, time.time() - end))
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.LR_decay
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * args.LR_decay

        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for inputs_val, labels_val in validation_loader:
        #         inputs_val = inputs_val.to(device)
        #         labels_val = labels_val.to(device)
        #         outputs_val = model(inputs_val)
        #         _, predicted = torch.max(outputs_val.data, 1)
        #         total += labels_val.size(0)
        #         correct += (predicted == labels_val).sum().item()
        #
        #         # print statistics
        #     print('Accuracy of the network on the %d validation examples: %.2f %%' % (validation_loader.dataset.examples.shape[0],
        #             100 * correct / total))
        #     if best_accuracy < (correct / total):
        #         torch.save(model.state_dict(), os.path.join(saving_path, 'best_model_parameters.pth'))
        #         save_anneal_func(model, paths_to_save)
        #         best_accuracy = correct/total
        #
        # if args.visu_progress:
        #     # accuracy_graph.append((correct / total))
        #     # rho_graph.append(rho)
        #     loss_for_epoch.append((running_loss_graph/number_of_batchs))
        #     x_values.append(next(index))
        #     # ax1 = plt.subplot(1, 2, 1)
        #     # ax1.plot(x_values, rho_graph, label='accuracy')
        #     # ax1.set_xlim([0, args.num_of_epochs])
        #     # ax1.set_ylim([0, 1])
        #     # ax1.set_title('Rho values')
        #     ax2 = plt.subplot(1, 2, 2)
        #     ax2.plot(x_values, loss_for_epoch, label='loss')
        #     ax2.set_xlim([0, args.num_of_epochs])
        #     ax2.set_title('Loss values')
        #     # plt.legend()
        #     plt.draw()
        #     if epoch == args.num_of_epochs - 1:
        #         plt.savefig(os.path.join(saving_path, 'progress plot.png'))
        #     # plt.pause(.3)
        #     plt.cla()

    return model
