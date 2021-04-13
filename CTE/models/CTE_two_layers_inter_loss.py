import Sandboxes.Farjon.CTE.layers._misc as torch_CTE_layers
import torch.nn as nn
import torch
from Sandboxes.Farjon.CTE.utils.annealing_mechanism_functions import update_ambiguity_thresholds, update_anneal_state
from Sandboxes.Farjon.CTE.debug import debug_1


class CTE(nn.Module):
    def __init__(self, args, input_shape, device):
        super(CTE, self).__init__()

        self.word_calc1 = torch_CTE_layers.FernBitWord(args.Fern1['M'], args.Fern1['K'], args.Fern1['L'], device, args.layer1_fern_path, args)
        VT1_input_shape = [input_shape[0] - args.Fern1['L'] + 1,
                           input_shape[1] - args.Fern1['L'] + 1,
                           input_shape[2]
                           ]
        self.voting_table1 = torch_CTE_layers.FernSparseTable(args.Fern1['K'],
                                                              args.Fern1['M'],
                                                              args.ST1['Num_of_active_words'],
                                                              args.ST1['D_out'],
                                                              VT1_input_shape,
                                                              args.layer1_S2D_path,
                                                              args
                                                              )
        self.avg_pool_1 = torch.nn.AvgPool2d(kernel_size=args.AvgPool1['kernel_size'], stride=2)
        self.avg_pool_1_2 = torch.nn.AvgPool2d(kernel_size=args.AvgPool1_1['kernel_size'], stride=1)

        self.word_calc2 = torch_CTE_layers.FernBitWord(args.Fern2['M'], args.Fern2['K'], args.Fern2['L'], device, args.layer2_fern_path, args)
        VT2_input_shape = [VT1_input_shape[0] - args.AvgPool1['kernel_size'] + 1 - args.Fern2['L'] + 1,
                           VT1_input_shape[1] - args.AvgPool1['kernel_size'] + 1 - args.Fern2['L'] + 1,
                           input_shape[2]
                           ]
        self.voting_table2 = torch_CTE_layers.FernSparseTable(args.Fern2['K'],
                                                              args.Fern2['M'],
                                                              args.ST2['Num_of_active_words'],
                                                              args.ST2['D_out'],
                                                              VT2_input_shape,
                                                              args.layer2_S2D_path,
                                                              args
                                                              )
        self.avg_pool_2 = torch.nn.AvgPool2d(kernel_size=args.AvgPool2['kernel_size'], stride=1)

        self.number_of_layers = args.number_of_layers
        self.args = args

    def forward(self, x):
        x = self.word_calc1(x)
        self.save_first_fern_values = None
        if self.word_calc1.anneal_state_params['count_till_update'] == self.word_calc1.anneal_state_params['batch_till_update']:
            self.save_first_fern_values = self.save_fern_values(self.word_calc1.bit_functions_values)
        if self.args.debug:
            IT, AT, B, x = self.voting_table1(x)
            # debug_1.debug_fern_AT_IT(AT, IT, B, self.word_calc1.ambiguity_thresholds)
        else:
            x = self.voting_table1(x)

        x1 = self.avg_pool_1_2(x)
        x1 = x1.view(x1.size(0), -1)
        x = self.avg_pool_1(x)

        x = self.word_calc2(x)
        self.save_second_fern_values = None
        if self.word_calc2.anneal_state_params['count_till_update'] == self.word_calc2.anneal_state_params['batch_till_update']:
            self.save_second_fern_values = self.save_fern_values(self.word_calc2.bit_functions_values)

        if self.args.debug:
            IT, AT, B, x = self.voting_table2(x)
        else:
            x = self.voting_table2(x)

        x = self.avg_pool_2(x)
        x = x.view(x.size(0), -1)
        return x, x1

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

    def on_batch_ends(self, update_layers):
        '''
        call back for the model - when batch ends, offsets are limited so they wont exceed the patch size, annealing mechanism
        parameters are updated (ambiguity thresholds and Rho)
        :param update_layers - a boolean list, length is the number of layers, each cell stated if the layer should be updated
        :return:
        '''

        if update_layers[0] == 1:
            self.word_calc1.limit_offsets_size()
            self.word_calc1.anneal_state_params, self.word_calc1.ambiguity_thresholds = update_ambiguity_thresholds(
                self.word_calc1.anneal_state_params,
                self.word_calc1.ambiguity_thresholds,
                self.save_first_fern_values
            )
            self.word_calc1.anneal_state_params = update_anneal_state(self.word_calc1.anneal_state_params)
        if update_layers[1] == 1:
            self.word_calc2.limit_offsets_size()
            self.word_calc2.anneal_state_params, self.word_calc2.ambiguity_thresholds = update_ambiguity_thresholds(
                        self.word_calc2.anneal_state_params,
                        self.word_calc2.ambiguity_thresholds,
                        self.save_second_fern_values
                    )

            self.word_calc2.anneal_state_params = update_anneal_state(self.word_calc2.anneal_state_params)

