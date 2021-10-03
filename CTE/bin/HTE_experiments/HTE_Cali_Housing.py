import os
import torch
import torch.nn as nn
from torch import optim
from CTE.utils.help_funcs import save_anneal_params, load_anneal_params, print_end_experiment_report, print_final_results
from CTE.bin.HTE_experiments.training_functions import train_loop
from CTE.bin.HTE_experiments.evaluation_function import eval_loop
from datetime import datetime

def Train_Cali_Housing(args, train_loader, test_loader, device, val_loader = None):

    # Letter recognition dataset has 16 features and 26 classes
    features_in = 8
    D_in = [features_in]
    D_out = []
    for i in range(args.num_of_layers - 1):
        if args.res_connection == 1:
            D_out.append(features_in)
            D_in.append(D_out[i])
        elif args.res_connection == 2:
            D_out.append(40)
            D_in.append(D_out[i]+D_in[0])
    D_out.append(1)

    # Decide on the ferns parameters and sparse table parameters
    # Fern parameters should include:
    #   K - number bit functions
    #   M - number of ferns
    #   L - patch size
    # Sparse Table should include:
    #   D_out - number of features for next layer
    K = args.number_of_BF
    M = args.num_of_ferns

    # define layers parameters
    args.Fern_layer, args.ST_layer = [], []
    for i in range(args.num_of_layers):
        layer_d_in = D_in[i]
        args.Fern_layer.append({'K': K, 'M': M, 'num_of_features': layer_d_in})
        layer_d_out = D_out[i]
        args.ST_layer.append({'Num_of_active_words': 2**(K), 'D_out': layer_d_out})

    # model parameters
    args.prune_type = 1
    args.input_size = [args.batch_size, D_in]

    if args.loss == 'MSE':
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss == 'uncertinty_yarin_gal':
        def mu_sig_loss(outputs, labels):
            y_pred = outputs[0]
            log_var = outputs[1]
            loss = torch.exp(-log_var) * torch.pow(y_pred - labels, 2) + log_var

            return loss
        criterion = mu_sig_loss

    args.number_of_batches = train_loader.dataset.examples.shape[0] / args.batch_size
    from CTE.models.HTE_reg import HTE

    model = HTE(args, args.input_size, device)

    # set learning rate to model parameters, we set different learning rate to W and V
    voting_table_LR_params_list = []
    for i in range(args.num_of_layers):
        voting_table_LR_params_list.append('voting_table_layers.' + str(i) + '.weights')
        voting_table_LR_params_list.append('voting_table.layers.' + str(i) + '.bias')

    voting_table_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in voting_table_LR_params_list, model.named_parameters()))))
    word_calc_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in voting_table_LR_params_list, model.named_parameters()))))

    if args.optimizer == 'ADAM':
        optimizer = optim.Adam([{'params': word_calc_params},
                               {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                                  lr=args.word_calc_learning_rate,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD([{'params': word_calc_params},
                               {'params': voting_table_params, 'lr': args.voting_table_learning_rate}],
                                  lr=args.word_calc_learning_rate,
                              weight_decay=args.weight_decay)
    else:
        assert 'no such optimizer, use only ADAM or sgd'

    # set path to save annealing mechanism parameters for checkpoint recovery
    checkpoint_paths_anneal_params = []
    args.checkpoint_folder_path = os.path.join(args.save_path, 'check_point')
    os.makedirs(args.checkpoint_folder_path, exist_ok=True)
    args.checkpoint_model_path = os.path.join(args.save_path, 'check_point', 'checkpoint_model.pt')
    for i in range(0, args.num_of_layers * 2, 2):
        checkpoint_paths_anneal_params.append(
            os.path.join(args.checkpoint_folder_path, 'ambiguity_thresholds_layer_' + str(i) + '.p'))
        checkpoint_paths_anneal_params.append(
            os.path.join(args.checkpoint_folder_path, 'anneal_params_' + str(i + 1) + '.p'))
    args.checkpoint_paths_anneal_params = checkpoint_paths_anneal_params

    # set path to save best model parameters and annealing parameters
    if val_loader is not None:
        val_anneal_params = []
        args.val_folder_path = os.path.join(args.save_path, 'best_val_model')
        os.makedirs(args.val_folder_path, exist_ok=True)
        args.val_model_path = os.path.join(args.val_folder_path, 'best_model.pt')
        for i in range(0, args.num_of_layers * 2, 2):
            val_anneal_params.append(
                os.path.join(args.val_folder_path, 'ambiguity_thresholds_layer_' + str(i) + '.p'))
            val_anneal_params.append(
                os.path.join(args.val_folder_path, 'anneal_params_' + str(i + 1) + '.p'))
        args.val_paths_anneal_params = val_anneal_params

    def save_model_anneal_params(model, paths_to_load):
        for i in range(0, args.num_of_layers * 2, 2):
            path_to_AT = paths_to_load[i]
            path_to_anneal_params = paths_to_load[i + 1]
            save_anneal_params(model.word_calc_layers[int(i / 2)], path_to_AT, path_to_anneal_params)

    def load_model_anneal_params(model, paths_to_save):
        for i in range(0, args.num_of_layers * 2, 2):
            path_to_AT = paths_to_save[i]
            path_to_anneal_params = paths_to_save[i + 1]
            AT, AP = load_anneal_params(path_to_AT, path_to_anneal_params, device)
            model.word_calc_layers[int(i / 2)].ambiguity_thresholds = AT
            model.word_calc_layers[int(i / 2)].anneal_state_params = AP
        return model

    # add lr scheduler
    if args.use_cosine_lr:
        update_lr = 10  # T_max
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, update_lr, eta_min=1e-4)
    else:
        scheduler = None

    final_model = train_loop(model,
                             train_loader,
                             val_loader,
                             criterion,
                             optimizer,
                             scheduler,
                             args,
                             save_model_anneal_params,
                             load_model_anneal_params,
                             )
    # save model and ambiguity_thresholds
    torch.save(final_model.state_dict(), os.path.join(args.save_path, 'final_model_parameters.pth'))

    final_paths_anneal_params = []
    for i in range(0, args.num_of_layers * 2, 2):
        final_paths_anneal_params.append(
            os.path.join(args.save_path, 'final_ambiguity_thresholds_layer_' + str(i) + '.p'))
        final_paths_anneal_params.append(os.path.join(args.save_path, 'final_anneal_params_' + str(i + 1) + '.p'))
    save_model_anneal_params(final_model, final_paths_anneal_params)

    final_accuracy = eval_loop(test_loader, final_model, device, args)
    print(f'final model accuracy is {final_accuracy}')

    path_to_parameters_save = os.path.join(args.save_path, 'final_parameters_values.csv')
    path_to_hyper_parameters_save = os.path.join(args.save_path, 'final_hyper_parameters_values.csv')
    print_end_experiment_report(args, final_model, optimizer,
                                (final_accuracy), test_loader.dataset.examples.shape[0],
                                path_to_parameters_save,
                                path_to_hyper_parameters_save)

    best_model_path = args.val_model_path
    best_model_anneal_params = args.val_paths_anneal_params
    best_model = final_model

    best_model_params = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_model_params['model_state_dict'])
    optimizer.load_state_dict(best_model_params['optimizer_state_dict'])
    load_model_anneal_params(best_model, best_model_anneal_params)

    best_accuracy = eval_loop(test_loader, best_model, device, args)
    print(f'best model accuracy is {best_accuracy}')

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    print('Current Timestamp : ', timestampStr)

    # print_results_to_csv
    print_final_results(args, best_accuracy, final_accuracy)
    return