import sys
import numpy as np
from CTE.layers._misc import FernBitWord_tabular, FernSparseTable_tabular
import torch.nn as nn
import torch
from CTE.utils.annealing_mechanism_functions import update_ambiguity_thresholds_tabular, update_Rho_tempature_tabular


class HTE(nn.Module):
    def __init__(self, args, input_shape, device):
        super(HTE, self).__init__()

        self.word_calc_layers = nn.ModuleList()
        self.voting_table_layers = nn.ModuleList()
        for i in range(args.number_of_layers):
            self.word_calc_layers.append(
                FernBitWord_tabular(
                    args.Fern_layer[i]['M'],
                    args.Fern_layer[i]['K'],
                    args.Fern_layer[i]['num_of_features'],
                    device,
                    args
                )
            )
            VT_input_shape = [input_shape[0], args.Fern_layer[i]['M'], args.Fern_layer[i]['K']]
            self.voting_table_layers.append(
                FernSparseTable_tabular(
                    args.Fern_layer[i]['K'],
                    args.Fern_layer[i]['M'],
                    args.ST_layer[i]['Num_of_active_words'],
                    args.ST_layer[i]['D_out'],
                    VT_input_shape,
                    args.prune_type,
                    device
                )
            )

        self.number_of_layers = args.number_of_layers
        self.args = args
        self.save_fern_bit_values = [None] * args.number_of_layers

        # self.word_calc1 = FernBitWord_tabular(args.Fern1['M'], args.Fern1['K'], args.Fern1['num_of_features'], args, device)
        # VT1_input_shape = [input_shape[0], args.Fern1['M'], args.Fern1['K']]
        # self.voting_table1 = FernSparseTable_tabular(args.Fern1['K'],
        #                                                       args.Fern1['M'],
        #                                                       args.ST1['Num_of_active_words'],
        #                                                       args.ST1['D_out'],
        #                                                       VT1_input_shape,
        #                                                       args.prune_type,
        #                                                       device,
        #                                                       args
        #                                                       )

    def forward(self, x):

        for i in range(self.number_of_layers):
            x = self.word_calc_layers[i](x)
            if self.word_calc_layers[i].anneal_state_params['count_till_update'] == self.word_calc_layers[i].anneal_state_params['batch_till_update']:
                self.save_fern_bit_values[i] = self.save_fern_values(self.word_calc_layers[i].bit_functions_values)
            x = self.voting_table_layers[i](x)

        # x = self.word_calc1(x)
        # self.save_first_fern_values = None
        # if self.word_calc1.anneal_state_params['count_till_update'] == self.word_calc1.anneal_state_params['batch_till_update']:
        #     self.save_first_fern_values = self.save_fern_values(self.word_calc1.bit_functions_values)
        # x = self.voting_table1(x)
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
        for i in range(self.number_of_layers):
            self.word_calc_layers[i].anneal_state_params, self.word_calc_layers[i].ambiguity_thresholds = update_ambiguity_thresholds_tabular(
                    self.word_calc_layers[i].anneal_state_params,
                    self.word_calc_layers[i].ambiguity_thresholds,
                    self.save_fern_bit_values[i],
                    device
            )
            self.word_calc_layers[i].anneal_state_params = update_Rho_tempature_tabular(self.word_calc_layers[i].anneal_state_params)

