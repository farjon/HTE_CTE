import os
import time
import torch
from GetEnvVar import GetEnvVar
import matplotlib.pyplot as plt
from itertools import count
from CTE.bin.HTE_experiments.evaluation_function import eval_loop
from tqdm import tqdm

def train_loop(model,
               train_loader,
               val_loader,
               criterion,
               optimizer,
               scheduler,
               args,
               save_anneal_func,
               load_model_func
               ):

    device = args.device
    # check if there is a checkpoint
    if os.path.isfile(args.checkpoint_model_path):
        checkpoint = torch.load(args.checkpoint_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        epoch_start = checkpoint['epoch'] + 1
        load_model_func(model, args.checkpoint_paths_anneal_params)
    else:
        epoch_start = 0

    model.to(device)
    model.train(True)
    # plotting variables
    x_values = []
    loss_for_epoch = []
    index = count()
    number_of_batchs = args.number_of_batches
    best_accuracy = 0
    best_epoch = 0
    for epoch in range(epoch_start, args.num_of_epochs):
        running_loss_graph = 0.0
        end = time.time()
        loop = tqdm(enumerate(train_loader), total = len(train_loader), leave=False )
        for batch_idx, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients - this method was reported as a faster way to zero grad
            for param in model.parameters():
                param.grad = None
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss_graph += float(loss.item())
            loss.backward()
            optimizer.step()
            model.on_batch_ends(device)
            loop.set_description(f'Epoch [{epoch+1}/{args.num_of_epochs}]')
            loop.set_postfix(loss=loss.item())
        avg_loss = (running_loss_graph / number_of_batchs)
        # print('epoch %d/%d, loss: %.3f,  time: %.3f' %
        #       (epoch + 1, args.num_of_epochs, avg_loss, time.time() - end))
        # learning rate step - using LR_decay or scheduler
        if scheduler is None:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * args.LR_decay
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] * args.LR_decay
        else:
            scheduler.step()
        #checkpoint save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, args.checkpoint_model_path)
        save_anneal_func(model, args.checkpoint_paths_anneal_params)

        if val_loader is not None:
            # if epoch >= args.end_rho_at_epoch:
            if epoch >= 0:
                accuracy = eval_loop(val_loader, model, device)
                if accuracy >= best_accuracy:
                    best_epoch = epoch
                    best_accuracy = accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, args.val_model_path)
                    save_anneal_func(model, args.val_paths_anneal_params)
                    model.train(True)
                    print(f'best validation accuracy is now {best_accuracy}')

        if args.visu_progress:
            loss_for_epoch.append(avg_loss)
            x_values.append(next(index))
            plt.plot(x_values, loss_for_epoch, label='loss')
            plt.xlim([epoch_start, args.num_of_epochs])
            plt.title('Loss values')
            plt.draw()
            # plt.pause(.3)
            if epoch == args.num_of_epochs - 1:
                plt.savefig(os.path.join(args.save_path, 'progress plot.png'))
            else:
                plt.savefig(os.path.join(args.checkpoint_folder_path, 'progress plot.png'))
            plt.cla()
    if epoch > 0:
        print(f'best results is {best_accuracy} at epoch {best_epoch}')
    return model
