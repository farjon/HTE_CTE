import pickle
import csv
import numpy as np

def save_anneal_params(layer, path_to_AT, path_to_anneal_params):
    '''

    :param: layer
    :param: path_to_AT
    :param: path_to_anneal_params
    '''

    layer_ambiguity_thresholds = layer.ambiguity_thresholds

    with open(path_to_AT, 'wb') as fp:
        pickle.dump(layer_ambiguity_thresholds, fp, protocol=pickle.HIGHEST_PROTOCOL)


    layer_anneal_params = layer.anneal_state_params

    with open(path_to_anneal_params, 'wb') as fp:
        pickle.dump(layer_anneal_params, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_anneal_params(path_to_AT, path_to_anneal_params):
    '''

    :param path_to_AT:
    :param path_to_anneal_params:
    :return:
    '''

    with open(path_to_AT, 'rb') as fp:
        layer_ambiguity_thresholds = pickle.load(fp)

    with open(path_to_anneal_params, 'rb') as fp:
        layer_anneal_params = pickle.load(fp)


    return layer_ambiguity_thresholds, layer_anneal_params

def print_end_experiment_report(args, model):
    path_to_save_file = args.path_to_parameters_save
    csv_file = open(path_to_save_file, "w")
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
    header = ['Layer', 'Fern', 'Bit-F', 'alpha-num', 'alpha-val', 'thresh', 'AT']
    csv_writer.writerow(header)
    for layer in range(args.number_of_layers):
        for M in range(args.Fern_layer[layer]['M']):
            for K in range(args.Fern_layer[layer]['K']):
                l_str = str(layer)
                current_AT = model.word_calc_layers._modules[l_str].ambiguity_thresholds[M][0][K].detach().cpu().numpy()
                current_thresh = model.word_calc_layers._modules[l_str].th[M][K].detach().cpu().numpy()
                current_alpha_val = np.max(model.word_calc_layers._modules[l_str].alpha[M][K].detach().cpu().numpy())
                current_alpha_num = np.argmax(model.word_calc_layers._modules[l_str].alpha[M][K].detach().cpu().numpy())
                csv_writer.writerow([layer, M, K, current_alpha_num, current_alpha_val, current_thresh, current_AT])
    csv_file.close()