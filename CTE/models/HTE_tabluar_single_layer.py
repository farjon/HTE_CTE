import sys
import numpy as np
import Sandboxes.Farjon.CTE.layers._misc as torch_CTE_layers
import torch.nn as nn
import torch
from Sandboxes.Farjon.CTE.utils.annealing_mechanism_functions import update_ambiguity_thresholds_tabular, update_Rho_tempature_tabular
from Sandboxes.Farjon.CTE.debug import debug_1
# from pytorch_memlab import profile


class HTE(nn.Module):
    def __init__(self, args, input_shape, device):
        super(HTE, self).__init__()

        self.word_calc1 = torch_CTE_layers.FernBitWord_tabular(args.Fern1['M'], args.Fern1['K'], args.Fern1['num_of_features'], args, device)
        VT1_input_shape = [input_shape[0], args.Fern1['M'], args.Fern1['K']]
        self.voting_table1 = torch_CTE_layers.FernSparseTable_tabular(args.Fern1['K'],
                                                              args.Fern1['M'],
                                                              args.ST1['Num_of_active_words'],
                                                              args.ST1['D_out'],
                                                              VT1_input_shape,
                                                              args.prune_type,
                                                              device,
                                                              args
                                                              )
        self.number_of_layers = args.number_of_layer
        self.args = args

    def forward(self, x):
        x = self.word_calc1(x)
        self.save_first_fern_values = None
        if self.word_calc1.anneal_state_params['count_till_update'] == self.word_calc1.anneal_state_params['batch_till_update']:
            self.save_first_fern_values = self.save_fern_values(self.word_calc1.bit_functions_values)
        x = self.voting_table1(x)
        return x

    def save_fern_values(self, bit_functions_values_list):
        '''
        saves the bit functions values for all positions and all images, for each fern. This function saves a specific layer
        bit function values. For each layer, the values are stored in a list of size (1, M) with M being the number of ferns.
        Each cell in the list contains (K, (N*H*M)) values.
        :param bit_functions_values_list: a list containing 2D tensor of size (K, N*H*W) containing the bit functions values
        '''

        new_list = []
        for bit_values in bit_functions_values_list:
            new_list.append(bit_values.data.clone())
        return new_list

    def on_batch_ends(self, device):
        '''
        call back for the model - when batch ends, offsets are limited so they wont exceed the patch size, annealing mechanism
        parameters are updated (ambiguity thresholds and Rho)
        :return:
        '''
        self.word_calc1.anneal_state_params, self.word_calc1.ambiguity_thresholds = update_ambiguity_thresholds_tabular(
                    self.word_calc1.anneal_state_params,
                    self.word_calc1.ambiguity_thresholds,
                    self.save_first_fern_values,
                    device
                )

        self.word_calc1.anneal_state_params = update_Rho_tempature_tabular(self.word_calc1.anneal_state_params)

