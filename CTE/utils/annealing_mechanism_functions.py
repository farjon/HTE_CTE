import torch
import numpy as np

def init_ambiguity_thresholds(num_of_ferns, K, device):
    '''
    Creates an ambiguity_thresholds list  of 1*num_of_ferns with all ambiguity fields
    :param num_of_ferns: number of ferns of a single CTE layer
    :param K: number of bit functions
    :return: ambiguity_thresholds: tensor of size (2,K) filled is zeros
    '''
    ambiguity_thresholds = []
    for i in range(num_of_ferns):
        pos_vector = torch.zeros([1, K], dtype = torch.float32).to(device)
        neg_vector = torch.zeros([1, K], dtype=torch.float32).to(device)
        ambiguity_thresholds.append(torch.cat([pos_vector, neg_vector], dim=0))

    return ambiguity_thresholds

def init_anneal_state(anneal_state_params = None):
    '''
    initializing the annealing mechanism parameters, if a parameter (or all) is missing in the input, defualt values are assigned
    :param anneal_state_params: dictionary holding the annealing mechanism parameters. If none - defualt valeus will be assigned to each parameter.
    :return: anneal_state_params
    '''
    if anneal_state_params is None:
        anneal_state_params = {}

    if 'sample_size' not in anneal_state_params:
        anneal_state_params['sample_size'] = 10000
    if 'Rho' not in anneal_state_params:
        anneal_state_params['Rho'] = 0.7
    if 'batch_till_update' not in anneal_state_params:
        anneal_state_params['batch_till_update'] = 0
    if 'count_till_update' not in anneal_state_params:
        anneal_state_params['count_till_update'] = 0
    # this parameter is the momentum for Rho, it is set to 0 at first for the first epoch
    if 'prev_ambiguity_th_weight' not in anneal_state_params:
        anneal_state_params['prev_ambiguity_th_weight'] = 0
    if 'use_sign_condition' not in anneal_state_params:
        anneal_state_params['use_sign_condition'] = True
    # calculated for batch_size = 200 and Rho starting at 0.7 for last epoch to be at 0.05 (after 70 epochs)
    if 'cooling_rate' not in anneal_state_params:
        anneal_state_params['cooling_rate'] = 0.99985
    if 'use_one_thresh' not in anneal_state_params:
        anneal_state_params['use_one_thresh'] = 1


    return anneal_state_params

def update_anneal_state(anneal_state_params):
    '''
    Updates the anneal_state_parameter 'Rho' for a CTE layer
    :param anneal_state_params: parameters controlling the ambiguity threshold dynamics
    :return: anneal_state_params: the updated parameters
    '''

    anneal_state_params['Rho'] = anneal_state_params['Rho'] * anneal_state_params['cooling_rate']

    return anneal_state_params

def update_ambiguity_thresholds(anneal_state_params, ambiguity_thresholds, bit_function_values):
    '''

    :param anneal_state_params:
    :param ambiguity_thresholds:
    :param bit_function_values: a list of size (1, number_of_ferns) each holding a tensor of size (K, N*H*W)
            containing the bit functions values for image and location
    :return:
    '''
    if anneal_state_params['count_till_update'] < anneal_state_params['batch_till_update']:
        anneal_state_params['count_till_update'] += 1
        anneal_state_params['prev_ambiguity_th_weight'] = 0.95
    else:
        anneal_state_params['batch_till_update'] = 10
        anneal_state_params['count_till_update'] = 0
        number_of_ferns = len(bit_function_values)

        current_ambiguity_th = []

        for fern in range(number_of_ferns):
            K = bit_function_values[fern].size(0)
            num_of_values = bit_function_values[fern].size(1)

            Rho = anneal_state_params['Rho'] * torch.ones(1, K).cuda()

            num_of_samples = torch.min(torch.tensor([anneal_state_params['sample_size'], num_of_values])).cuda()

            if anneal_state_params['use_one_thresh'] == 0:

                random_sample_indices = (torch.randperm(num_of_values).cuda())[:num_of_samples]
                B_values, B_indices = torch.sort(bit_function_values[fern][:,random_sample_indices])

                current_ambiguity_th.append(torch.zeros(2, K).cuda())

                for bit_function in range(K):
                    if Rho[0,bit_function] == 0:
                        continue
                    B_bit_function_len = B_values[bit_function, :].size(0)
                    positive_value_indices = torch.where(B_values[bit_function, :] > 0, torch.ones(B_bit_function_len).cuda(), torch.zeros(B_bit_function_len).cuda()).nonzero()

                    if (positive_value_indices.size(0) == 0):
                        current_ambiguity_th[fern][0, bit_function] = float('nan')
                        first_positive_value_index = num_of_samples
                    else:
                        first_positive_value_index = positive_value_indices[0]
                        if anneal_state_params['use_sign_condition']:
                            B_positive = B_values[bit_function, first_positive_value_index:]
                            num_of_patches_in_half_rho = torch.ceil(Rho[0, bit_function] * B_positive.nelement())
                            current_ambiguity_th[fern][0, bit_function] = B_positive[torch._cast_Int(torch.min(torch.tensor([num_of_patches_in_half_rho.item(), B_positive.nelement()])) - 1)]
                        else:
                            #TODO - debug this option
                            num_of_patches_in_half_rho = torch.ceil(Rho[0, bit_function] / 2 * num_of_samples)
                            current_ambiguity_th[fern][0, bit_function] = B_values[bit_function,
                                                                            torch.min(first_positive_value_index + num_of_patches_in_half_rho - 1, num_of_samples)]

                    if first_positive_value_index == 0:
                        current_ambiguity_th[fern][1, bit_function] = float('nan')
                    else:
                        if anneal_state_params['use_sign_condition']:
                            B_negative = B_values[bit_function, :first_positive_value_index]
                            num_of_patches_in_half_rho = torch.ceil(Rho[0, bit_function] * B_negative.nelement())
                            current_ambiguity_th[fern][1, bit_function] = B_negative[torch._cast_Int(torch.max(torch.tensor([B_negative.nelement() - num_of_patches_in_half_rho, 0])))]
                        else:
                            # TODO - debug this option
                            num_of_patches_in_half_rho = torch.ceil(Rho(bit_function) / 2 * num_of_samples)
                            current_ambiguity_th[fern][1, bit_function] = B_negative[bit_function,
                                                                                     torch.max(first_positive_value_index - num_of_patches_in_half_rho, 1)]

                    if torch.isnan(current_ambiguity_th[fern][0, bit_function]).item():
                        current_ambiguity_th[fern][0, bit_function] = -current_ambiguity_th[fern][1, bit_function]
                        # print('Bit lost meaning')

                    if torch.isnan(current_ambiguity_th[fern][1, bit_function]).item():
                        current_ambiguity_th[fern][1, bit_function] = -current_ambiguity_th[fern][0, bit_function]
                        # print('Bit lost meaning')

            else:
                random_sample_indices = (torch.randperm(num_of_values).cuda())[:num_of_samples]
                B_values, B_indices = torch.sort(torch.abs(bit_function_values[fern][:, random_sample_indices]))

                current_ambiguity_th.append(torch.zeros(2, K).cuda())
                for bit_function in range(K):
                    if Rho[0, bit_function] == 0:
                        continue
                    Rho_percentile = torch._cast_Int(Rho[0, bit_function] * B_values[bit_function, :].size(0))
                    threshold = B_values[bit_function, Rho_percentile] + 1e-6
                    if (threshold > 0):
                        current_ambiguity_th[fern][0, bit_function] = threshold # positive threshold
                        current_ambiguity_th[fern][1, bit_function] = -threshold # negative threshold
                    else:
                        current_ambiguity_th[fern][0, bit_function] = -threshold  # positive threshold
                        current_ambiguity_th[fern][1, bit_function] = threshold  # negative threshold

            ambiguity_thresholds[fern] = (anneal_state_params['prev_ambiguity_th_weight'] * ambiguity_thresholds[fern]
                                          + (1 - anneal_state_params['prev_ambiguity_th_weight']) *
                                          current_ambiguity_th[fern])

    return anneal_state_params, ambiguity_thresholds

def init_anneal_state_tabular(args, anneal_state_params = None):
    '''
    initializing the annealing mechanism parameters, if a parameter (or all) is missing in the input, defualt values are assigned
    :param args: arguments of the training procedure
    :param anneal_state_params: dictionary holding the annealing mechanism parameters. If none - defualt valeus will be assigned to each parameter.
    :return: anneal_state_params
    '''
    if anneal_state_params is None:
        anneal_state_params = {}

    if 'sample_size' not in anneal_state_params:
        anneal_state_params['sample_size'] = 1000
    if 'Rho' not in anneal_state_params:
        anneal_state_params['Rho'] = 0.7
    if 'batch_till_update' not in anneal_state_params:
        anneal_state_params['batch_till_update'] = 0
    if 'count_till_update' not in anneal_state_params:
        anneal_state_params['count_till_update'] = 0
    # this parameter is the momentum for Rho, it is set to 0 at first for the first epoch
    if 'prev_ambiguity_th_weight' not in anneal_state_params:
        anneal_state_params['prev_ambiguity_th_weight'] = 0
    if 'use_sign_condition' not in anneal_state_params:
        anneal_state_params['use_sign_condition'] = True
    # rho = 0 is set to 0.00001, the model will run with hard ferns for 3 epochs
    if 'cooling_rate' not in anneal_state_params:
        anneal_state_params['cooling_rate'] = (0.00001/anneal_state_params['Rho'])**(1/((args.num_of_epochs-3)*args.number_of_batches))
    if 'tempature' not in anneal_state_params:
        anneal_state_params['tempature'] = 1
    if 'tempature_heat_rate' not in anneal_state_params:
        anneal_state_params['tempature_heat_rate'] = 1.0005
    return anneal_state_params

def update_Rho_tempature_tabular(anneal_state_params):
    '''
    Updates the anneal_state_parameter 'Rho' for a CTE layer
    :param anneal_state_params: parameters controlling the ambiguity threshold dynamics
    :return: anneal_state_params: the updated parameters
    '''

    anneal_state_params['tempature'] = anneal_state_params['tempature'] * anneal_state_params['tempature_heat_rate']
    anneal_state_params['Rho'] = anneal_state_params['Rho'] * anneal_state_params['cooling_rate']
    return anneal_state_params

def update_ambiguity_thresholds_tabular(anneal_state_params, ambiguity_thresholds, bit_function_values, device):
    '''

    :param anneal_state_params:
    :param ambiguity_thresholds:
    :param bit_function_values: a list of size (1, number_of_ferns) each holding a tensor of size (K, N*H*W)
            containing the bit functions values for image and location
    :return:
    '''
    if anneal_state_params['count_till_update'] < anneal_state_params['batch_till_update']:
        anneal_state_params['count_till_update'] += 1
        anneal_state_params['prev_ambiguity_th_weight'] = 0.99
    else:
        anneal_state_params['batch_till_update'] = 1
        anneal_state_params['count_till_update'] = 0


        number_of_ferns = len(bit_function_values)

        current_ambiguity_th = []

        for fern in range(number_of_ferns):
            K = bit_function_values[fern].size(1)
            num_of_values = bit_function_values[fern].size(0)

            Rho = anneal_state_params['Rho'] * torch.ones(1, K).to(device)

            num_of_samples = torch.min(torch.tensor([anneal_state_params['sample_size'], num_of_values])).to(device)


            random_sample_indices = (torch.randperm(num_of_values).to(device))[:num_of_samples]
            B_values, B_indices = torch.sort(torch.abs(bit_function_values[fern][random_sample_indices, :]))

            current_ambiguity_th.append(torch.zeros(2, K).to(device))
            for bit_function in range(K):
                if Rho[0, bit_function] == 0:
                    continue
                Rho_percentile = torch._cast_Int(Rho[0, bit_function] * B_values[bit_function, :].size(0))
                threshold = B_values[bit_function, Rho_percentile] + 1e-6;
                if (threshold > 0):
                    current_ambiguity_th[fern][0, bit_function] = threshold # positive threshold
                    current_ambiguity_th[fern][1, bit_function] = -threshold # negative threshold
                else:
                    current_ambiguity_th[fern][0, bit_function] = -threshold  # positive threshold
                    current_ambiguity_th[fern][1, bit_function] = threshold  # negative threshold

            ambiguity_thresholds[fern] = (anneal_state_params['prev_ambiguity_th_weight'] * ambiguity_thresholds[fern]
                                          + (1 - anneal_state_params['prev_ambiguity_th_weight']) *
                                          current_ambiguity_th[fern])

    return anneal_state_params, ambiguity_thresholds