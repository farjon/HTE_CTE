import pickle
import csv
import numpy as np
import torch

def save_anneal_params(layer, path_to_AT, path_to_anneal_params):
    '''

    :param: layer
    :param: path_to_AT
    :param: path_to_anneal_params
    '''

    layer_ambiguity_thresholds = layer.ambiguity_thresholds
    torch.save(layer_ambiguity_thresholds, path_to_AT)

    layer_anneal_params = layer.anneal_state_params
    torch.save(layer_anneal_params, path_to_anneal_params)


def load_anneal_params(path_to_AT, path_to_anneal_params, device):
    '''

    :param path_to_AT:
    :param path_to_anneal_params:
    :return:
    '''

    layer_ambiguity_thresholds = torch.load(path_to_AT, map_location=device)
    layer_anneal_params = torch.load(path_to_anneal_params, map_location=device)

    return layer_ambiguity_thresholds, layer_anneal_params

def timer_decorator(func):
    import time
    def timer(*args, **kwargs):
        ts = time.time()
        results = func(*args, **kwargs)
        te = time.time()
        print(func, (te-ts))
        return results
    return timer

def print_end_experiment_report(args, model, optimizer, test_score, testset_size, path_to_parameters, path_to_hyper_params):
    # print model parameters
    path_to_save_file = path_to_parameters
    csv_file = open(path_to_save_file, "w")
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
    header = ['Layer', 'Fern', 'Bit-F', 'alpha-num', 'alpha-val_BS', 'alpha-val_AS', 'thresh', 'AT']
    csv_writer.writerow(header)
    for layer in range(args.num_of_layers):
        for M in range(args.Fern_layer[layer]['M']):
            for K in range(args.Fern_layer[layer]['K']):
                l_str = str(layer)
                current_AT = model.word_calc_layers._modules[l_str].ambiguity_thresholds[M][0][K].detach().cpu().numpy()
                current_thresh = model.word_calc_layers._modules[l_str].th[M][K].detach().cpu().numpy()
                current_alpha_val_BS = np.max(model.word_calc_layers._modules[l_str].alpha[M][K].detach().cpu().numpy())
                current_alpha_val_AS = np.max(model.word_calc_layers._modules[l_str].alpha_after_softmax[M][K])
                current_alpha_num = np.argmax(model.word_calc_layers._modules[l_str].alpha[M][K].detach().cpu().numpy())
                csv_writer.writerow([layer, M, K, current_alpha_num, current_alpha_val_BS, current_alpha_val_AS, current_thresh, current_AT])
    csv_file.close()

    # print model hyper-parameters
    path_to_save_file = path_to_hyper_params
    csv_file = open(path_to_save_file, "w")
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
    csv_writer.writerow(['number of layers', args.num_of_layers])
    csv_writer.writerow(['test accuracy', test_score])
    csv_writer.writerow(['test size', testset_size])
    csv_writer.writerow(['number of epochs', args.num_of_epochs])
    csv_writer.writerow(['batch size', args.batch_size])
    csv_writer.writerow(['optimizer', args.optimizer])
    csv_writer.writerow(['lr decay factor', args.LR_decay])
    csv_writer.writerow(['Rho cooling rate', model.word_calc_layers._modules['0'].anneal_state_params['cooling_rate']])
    csv_writer.writerow(['beta heat rate', model.word_calc_layers._modules['0'].anneal_state_params['tempature_heat_rate']])
    csv_writer.writerow(['AT prev weight', model.word_calc_layers._modules['0'].anneal_state_params['prev_ambiguity_th_weight']])
    csv_writer.writerow(['param', 'end value'])
    csv_writer.writerow(['beta', model.word_calc_layers._modules['0'].anneal_state_params['tempature']])
    csv_writer.writerow(['rho',  model.word_calc_layers._modules['0'].anneal_state_params['Rho']])
    csv_writer.writerow(['lr_W', optimizer.param_groups[0]['lr']])
    csv_writer.writerow(['lr_V', optimizer.param_groups[1]['lr']])
    csv_writer.writerow(['batch_norm', args.batch_norm])
    csv_writer.writerow(['cosine lr scheduler', args.use_cosine_lr])
    csv_file.close()

def print_final_results(args, best_res, final_res):
    exp_num = args.experiment_number
    num_of_ferns = args.Fern_layer[0]['M']
    number_of_BF = args.Fern_layer[0]['K']
    D_out = args.ST_layer[0]['D_out']
    num_of_layers = args.num_of_layers
    dataset = args.dataset_name
    row = f'{exp_num}, {dataset}, {num_of_ferns}, {number_of_BF}, {D_out}, {num_of_layers}, {best_res}, {final_res}'
    with open(args.results_csv_file, "a", newline='') as fd:
        fd.write(row)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
