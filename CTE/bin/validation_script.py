import torch
from itertools import count
import matplotlib.pyplot as plt


def validate_model(model, val_loader, debug):
    accuracy_graph = []
    loss_for_epoch = []
    x_values = []
    index = count()
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
        if debug:
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
        best_accuracy = correct_val / total_val