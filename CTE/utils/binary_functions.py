import torch


def q_bin_activation(features, weights, thresholds, ambiguity_thresholds):
    '''
    compute a complete ferns with K bit functions for all examples
    :param T: a matrix of size (N, D_in) where N is the number of examples,
        and D_in is the number of features
    :param alpha: a tensor of size (num_of_bit_functions, D_in) containing the weights for each cell of each of the bit functions in a given fern
    :param thresh: a tensor of size (num_of_bit_functions, D_in) containing the threshold of bit function
    :param ambiguity_thresholds: vector of size (K,2) holding the ambiguity thresholds (pos, neg)
    :return: B: a matrix of size (N, K) containing the bit function value (in the range [0,1]) for each exmaple
    :return: b: a 3D tensor of size (N, K) containing the bit function value before bounding it, for each example
    '''
    pos_ambiguity = ambiguity_thresholds[0]
    neg_ambiguity = -1 * ambiguity_thresholds[1]

    weighted_features = torch.einsum('ik,jk->ij', features, weights)
    weighted_features[torch.abs(weighted_features) < 1e-5] = 0
    # linear sigmoid function
    b = torch.sub(weighted_features, thresholds)
    unclipped_B = (b + neg_ambiguity) / (neg_ambiguity + pos_ambiguity + 10e-30)
    B = torch.clamp(unclipped_B, 0, 1)
    # B is bounded between [0,1], b is the real value of the bit function
    return B, b