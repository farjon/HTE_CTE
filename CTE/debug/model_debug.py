import sys
import numpy as np
import Farjon.CTE.debug.layers_debug as torch_CTE_layers
import torch.nn as nn
import torch
from annealing_mechanism_functions import update_ambiguity_thresholds


class CTE(nn.Module):
    def __init__(self, args, num_of_classes):
        super(CTE, self).__init__()

        constant_indices1 = create_constant_inds(args.input_size[0], args.input_size[1], args.batch_size)

        self.word_calc1 = torch_CTE_layers.FernBitWord(args.Fern1['M'], args.Fern1['K'], args.Fern1['L'])
        self.voting_table1 = torch_CTE_layers.FernSparseTable(constant_indices1, args.Fern1['K'], args.Fern1['M'],
                                                              args.ST1['Num_of_active_words'], args.ST1['D_out'])
        self.avg_pool_1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.pred = nn.Linear(args.ST1['D_out'] * int(args.input_size[0] / 2) * int(args.input_size[1] / 2),
                              num_of_classes, bias=True).to('cuda')

        self.number_of_layers = args.number_of_layer

    def forward(self, x):
        x = self.word_calc1(x)
        self.save_first_fern_values = self.word_calc1.bit_functions_values
        x = self.voting_table1(x)
        x = self.avg_pool_1(x)

        x = x.view(x.size(0), -1)
        x = self.pred(x)
        return x

    def save_fern_values(self, bit_functions_values):
        '''
        saves the bit functions values for all positions and all images, for each fern. This function saves a specific layer
        bit function values. For each layer, the values are stored in a list of size (1, M) with M being the number of ferns.
        Each cell in the list contains (K, (N*H*M)) values.
        :param bit_functions_values: a list containing 2D tensor of size (K, N*H*W) containing the bit functions values
        '''
        self.all_layers_bit_functions_values.append(bit_functions_values)

    def on_batch_ends(self):

        self.word_calc1.anneal_state_params, self.word_calc1.ambiguity_thresholds = update_ambiguity_thresholds(
            self.word_calc1.anneal_state_params,
            self.word_calc1.ambiguity_thresholds,
            self.save_first_fern_values
        )

        # TODO - limit offsets to patch size!!!


def create_constant_inds(H, W, num_of_images):
    '''

    :param H:
    :param W:
    :param num_of_images:
    :return:
    '''

    x_temp = np.linspace(0, W - 1, num=W, dtype='int')
    y_temp = np.linspace(0, H - 1, num=H, dtype='int')
    Xv, Yv = np.meshgrid(x_temp, y_temp)
    X = np.tile(Xv.flatten(), num_of_images)
    Y = np.tile(Yv.flatten(), num_of_images)
    first_col = np.repeat(np.arange(0, num_of_images), W * H)
    ones_vector = np.ones([W * H * num_of_images])
    constant_inds = np.vstack((first_col, ones_vector, Y, X))
    return constant_inds

