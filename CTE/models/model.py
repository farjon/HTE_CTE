import sys
import numpy as np
import Farjon.CTE.layers._misc as torch_CTE_layers
import torch.nn as nn
import torch
from annealing_mechanism_functions import update_ambiguity_thresholds, update_anneal_state
# from pytorch_memlab import profile


class CTE(nn.Module):
    def __init__(self, args, num_of_classes):
        super(CTE, self).__init__()

        self.word_calc1 = torch_CTE_layers.FernBitWord(args.Fern1['M'], args.Fern1['K'], args.Fern1['L'])
        self.voting_table1 = torch_CTE_layers.FernSparseTable(args.Fern1['K'], args.Fern1['M'], args.ST1['Num_of_active_words'], args.ST1['D_out'])
        self.avg_pool_1 = torch.nn.AvgPool2d(kernel_size=args.AvgPool1['kernel_size'], stride=1)

        self.word_calc2 = torch_CTE_layers.FernBitWord(args.Fern2['M'], args.Fern2['K'], args.Fern2['L'])
        self.voting_table2 = torch_CTE_layers.FernSparseTable(args.Fern2['K'], args.Fern2['M'], args.ST2['Num_of_active_words'], args.ST2['D_out'])
        self.avg_pool_2 = torch.nn.AvgPool2d(kernel_size=args.AvgPool2['kernel_size'], stride=1)

        self.number_of_layers = args.number_of_layer

    def forward(self, x):
        x = self.word_calc1(x)
        if self.word_calc1.anneal_state_params['count_till_update'] == self.word_calc1.anneal_state_params['batch_till_update']:
            self.save_first_fern_values = self.save_fern_values(self.word_calc1.bit_functions_values)
        x = self.voting_table1(x)
        x = self.avg_pool_1(x)

        x = self.word_calc2(x)
        if self.word_calc2.anneal_state_params['count_till_update'] == self.word_calc2.anneal_state_params['batch_till_update']:
            self.save_second_fern_values = self.save_fern_values(self.word_calc2.bit_functions_values)
        x = self.voting_table2(x)

        x = self.avg_pool_2(x)
        x = x.view(x.size(0), -1)
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

    def on_batch_ends(self):
        '''
        call back for the model - when batch ends, offsets are limited so they wont exceed the patch size, annealing mechanism
        parameters are updated (ambiguity thresholds and Rho)
        :return:
        '''
        self.word_calc1.limit_offsets_size()
        self.word_calc2.limit_offsets_size()

        self.word_calc1.anneal_state_params, self.word_calc1.ambiguity_thresholds = update_ambiguity_thresholds(
                    self.word_calc1.anneal_state_params,
                    self.word_calc1.ambiguity_thresholds,
                    self.save_first_fern_values
                )

        self.word_calc2.anneal_state_params, self.word_calc2.ambiguity_thresholds = update_ambiguity_thresholds(
                    self.word_calc2.anneal_state_params,
                    self.word_calc2.ambiguity_thresholds,
                    self.save_second_fern_values
                )

        self.word_calc1.anneal_state_params = update_anneal_state(self.word_calc1.anneal_state_params)
        self.word_calc2.anneal_state_params = update_anneal_state(self.word_calc2.anneal_state_params)


