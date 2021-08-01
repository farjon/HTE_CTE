import torch
from sklearn import metrics

def eval_loop(test_loader, model, device, args):
    correct = 0
    test_examples = test_loader.dataset.examples.shape[0]
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            y_true.extend(labels_test.detach().cpu().numpy().tolist())
            outputs_test = model(inputs_test)
            _, predicted = torch.max(outputs_test.data, 1)
            y_pred.extend(predicted.detach().cpu().numpy().tolist())
            correct += (predicted == labels_test).sum().item()
    if args.monitor_balanced_acc:
        balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
        print(f'the balanced accuracy is {balanced_accuracy}')
    if args.monitor_auc:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        print(f'the auc is {metrics.auc(fpr, tpr)}')

    return 100*(correct/test_examples)
