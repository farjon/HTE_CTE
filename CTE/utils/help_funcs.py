import pickle

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