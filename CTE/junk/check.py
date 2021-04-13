import Sandboxes.Farjon.CTE.layers._misc as torch_CTE_layers
import torch.nn as nn
import torch
from Sandboxes.Farjon.CTE.utils.annealing_mechanism_functions import update_ambiguity_thresholds, update_anneal_state


class CTE(nn.Module):
    def __init__(self, args, input_shape, device):
        super(CTE, self).__init__()
        self.word_calc = []
        self.voting_table = []
        self.avg_pool = []

        self.word_calc.append(torch_CTE_layers.FernBitWord(args.Fern1['M'], args.Fern1['K'], args.Fern1['L'], device, args.layer1_fern_path, args))

        VT1_input_shape = [input_shape[0] - args.Fern1['L'] + 1, input_shape[1] - args.Fern1['L'] + 1, input_shape[2]]


        self.voting_table.append(torch_CTE_layers.FernSparseTable(args.Fern1['K'], args.Fern1['M'], args.ST1['Num_of_active_words'], args.ST1['D_out'], VT1_input_shape, args.layer1_S2D_path, args))

        self.avg_pool.append(torch.nn.AvgPool2d(kernel_size=args.AvgPool1['kernel_size'], stride=1))

        self.number_of_layers = args.number_of_layers
        self.args = args
        self.fern_bit_values = list(range(args.number_of_layers))

    def forward(self, x):
        # the sequence is word_calc -> voting table -> avg pool
        for i in range(self.number_of_layers):
            x = self.word_calc[i](x)
            if self.word_calc[i].anneal_state_params['count_till_update'] == self.word_calc[i].anneal_state_params['batch_till_update']:
                self.fern_bit_values[i] = self.save_fern_values(self.word_calc[i].bit_functions_values)
            if self.args.debug:
                IT, AT, B, x = self.voting_table[i](x)
            else:
                x = self.voting_table[i](x)
            x = self.avg_pool[i](x)
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
        for i in range(self.number_of_layers):
            self.word_calc[i].limit_offsets_size()

            self.word_calc[i].anneal_state_params, self.word_calc[i].ambiguity_thresholds = update_ambiguity_thresholds(
                        self.word_calc[i].anneal_state_params,
                        self.word_calc[i].ambiguity_thresholds,
                        self.fern_bit_values[i]
                    )

            self.word_calc[i].anneal_state_params = update_anneal_state(self.word_calc[i].anneal_state_params)

