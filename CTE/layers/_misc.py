import os
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from CTE.utils.annealing_mechanism_functions import init_ambiguity_thresholds
from CTE.utils.annealing_mechanism_functions import init_anneal_state, init_anneal_state_tabular
# from pytorch_memlab import profile

# pytorch uses NxCxHxW dimension order!!

class FernBitWord(nn.Module):

    def __init__(self, num_of_ferns, K, Patch_Size, device, load_weights = None, args = None, padding = 'valid'):
        '''
        Constructor for the Fern bit word encoder.
        This layer includes the following learnable parameters:
            - dx1 - X offsets of the fist pixel for each bit function in each fern
            - dx2 - X offsets of the second pixel for each bit function in each fern
            - dy1 - Y offsets of the fist pixel for each bit function in each fern
            - dy2 - Y offsets of the second pixel for each bit function in each fern
            - th - the threshold for each of the bit function in each fern
        the length of each of these parameter lists is hence number_of_ferns*number_of_bit_functions (M*K)

        :param num_of_ferns: number of ferns to used in the layer
        :param K: number of bit functions in each fern
        :param Patch_Size: the size of the patch to encode
        :param load_weights: path to layer's parameters. If False, weights will be initialized randomly
        :param padding: can be either same or valid:
                        same - output image size will remain as input size.
                        valid - output size will decrease by (input_size - Patch_Size + 1)
        '''
        super(FernBitWord, self).__init__()
        self.padding = padding
        self.num_of_ferns = num_of_ferns
        self.num_of_bit_functions = K

        if load_weights is None:
            self.dx1 = nn.Parameter(torch.from_numpy(self.__initialize_ferns_parameters(num_of_ferns, K, Patch_Size)).cuda())
            self.dx2 = nn.Parameter(torch.from_numpy(self.__initialize_ferns_parameters(num_of_ferns, K, Patch_Size)).cuda())
            self.dy1 = nn.Parameter(torch.from_numpy(self.__initialize_ferns_parameters(num_of_ferns, K, Patch_Size)).cuda())
            self.dy2 = nn.Parameter(torch.from_numpy(self.__initialize_ferns_parameters(num_of_ferns, K, Patch_Size)).cuda())
            self.th = nn.Parameter(torch.from_numpy(self.__initialize_ferns_parameters(num_of_ferns, K, 0)).cuda())
        else:
            self.dx1 = nn.Parameter(torch.from_numpy(np.transpose(np.array(pd.read_csv(os.path.join(load_weights, 'dx1.csv'), header = None), dtype = np.single))).cuda())
            self.dx2 = nn.Parameter(torch.from_numpy(np.transpose(np.array(pd.read_csv(os.path.join(load_weights, 'dx2.csv'), header = None), dtype = np.single))).cuda())
            self.dy1 = nn.Parameter(torch.from_numpy(np.transpose(np.array(pd.read_csv(os.path.join(load_weights, 'dy1.csv'), header = None), dtype = np.single))).cuda())
            self.dy2 = nn.Parameter(torch.from_numpy(np.transpose(np.array(pd.read_csv(os.path.join(load_weights, 'dy2.csv'), header = None), dtype = np.single))).cuda())
            self.th = nn.Parameter(torch.from_numpy(np.transpose(np.array(pd.read_csv(os.path.join(load_weights, 'th.csv'), header = None), dtype = np.single))).cuda())

        # The following commeneted parameters are for debug purposes only
        # self.dx1 = nn.Parameter(torch.tensor([[0., 0 ,  0  , 0]]).cuda())
        # self.dy1 = nn.Parameter(torch.tensor([[0., 0 ,  0  , 0]]).cuda())
        # self.dx2 = nn.Parameter(torch.tensor([[9., -9, 9   , -9]]).cuda())
        # self.dy2 = nn.Parameter(torch.tensor([[0., 9 , -9  , -9]]).cuda())
        # self.th = nn.Parameter(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).cuda())

        channels_cycle = np.tile(np.arange(args.input_size[2]), int(np.ceil(K / args.input_size[2])))[:K]
        self.channels_to_slice = torch.from_numpy(channels_cycle).cuda()
        self.Patch_size = torch.tensor(Patch_Size).cuda()
        self.ambiguity_thresholds = init_ambiguity_thresholds(self.num_of_ferns, self.num_of_bit_functions, device)
        self.anneal_state_params = init_anneal_state()

    def forward(self, T):
        '''
        applied on the input tensor, this layer slices the tensor according to the number of bit function and the number of ferns
        and calculates the bit function (for now only comparison between 2 pixels and a threshold is implemented) for each of the
        ferns.
        :param x: a 4D tensor of size (N, D, H, W) where N is the number of images,
            D is the the number of channels, and H,W are the height and width
        :return: output: a 4D tensor of size (N, M*K, H, W) where M*K is the number_of_ferns*number_of_bit_functions.
            This tensor hold the bit function values for each pixels location in each of the images.
        '''


        # take parameters from input
        T_size = T.size()
        if self.padding == 'valid':
            output_H_size = T_size[2] - self.Patch_size + 1
            output_W_size = T_size[3] - self.Patch_size + 1
        elif self.padding == 'same':
            # local parameters
            dx1_max = torch.max(torch.max(torch.abs(self.dx1)))
            dx2_max = torch.max(torch.max(torch.abs(self.dx2)))
            dy1_max = torch.max(torch.max(torch.abs(self.dy1)))
            dy2_max = torch.max(torch.max(torch.abs(self.dy2)))
            current_patch_size = torch.max(torch.tensor([dx1_max, dx2_max, dy1_max, dy2_max]).cuda())
            output_H_size = T_size[2]
            output_W_size = T_size[3]

        Bits = torch.zeros((T_size[0], self.num_of_bit_functions * self.num_of_ferns, output_H_size, output_W_size)).cuda()

        bit_functions_values = []
        index = 0
        for m in range(self.num_of_ferns):
            # if self.anneal_state_params['count_till_update'] == self.anneal_state_params['batch_till_update']:
            fern_bit_function_values = torch.zeros(self.num_of_bit_functions, T_size[0]*output_H_size*output_W_size).cuda()
            for k in range(self.num_of_bit_functions):
                current_channel = self.channels_to_slice[k].long()
                channel = T[:, current_channel, :, :]

                # take W,H values of the network's input before padding
                H = T.size()[2]
                W = T.size()[3]
                if self.padding == 'valid':
                    # We need the padding since we used the four-pixels interpolation
                    pad_L = torch.tensor(0).cuda()
                    padding = nn.ZeroPad2d((0,1,0,1))
                    channel_padded = padding(channel)
                elif self.padding == 'same':
                    pad_L = torch.ceil(current_patch_size + 1)
                    #TODO - remove the need for .item()
                    padding = nn.ConstantPad2d(torch._cast_Int(pad_L).item(), 0)
                    channel_padded = padding(channel)
                Bits[:, index, :, :], bit_values = self.__Bit(channel_padded, self.dx1[m, k], self.dx2[m, k], self.dy1[m, k], self.dy2[m, k], self.th[m, k], pad_L, H, W, self.ambiguity_thresholds[m][:,k])
                # if self.anneal_state_params['count_till_update'] == self.anneal_state_params['batch_till_update']:
                fern_bit_function_values[k, :] = bit_values.view(-1)
                index = index + 1
            if self.anneal_state_params['count_till_update'] == self.anneal_state_params['batch_till_update']:
                bit_functions_values.append(fern_bit_function_values)
        # output = Bits
        # save the bit function values for the annealing mechanism
        self.bit_functions_values = bit_functions_values

        return Bits

    def __Bit(self, T, dx1, dx2, dy1, dy2, thresh, pad_L, H, W, ambiguity_thresholds):
        '''
        compute a single bit function of a specific fern for all images
        :param T: a 3D tensor of size (N, H, W) where N is the number of images,
            1 is the sliced channel, and H,W are the height and width
        :param dx1: the first pixel offset x-dim
        :param dy1: the first pixel offset y-dim
        :param dx2: the second pixel offset x-dim
        :param dy2: the second pixel offset y-dim
        :param thresh: the threshold of bit function (p1 - p2 - thresh)
        :param pad_L: the patch size (LxL)
        :param H: original tensor height before padding
        :param W: original tensor width before padding
        :param ambiguity_thresholds: vector of size (1,2) holding the ambiguity thresholds (pos, neg)
        :return: B: a 3D tensor of size (N, H, W) containing the bit function value (in the range [0,1]) for pixel location in each of the images
        :return: b: a 3D tensor of size (N, H, W) containing the bit function value before bounding it, for pixel location in each of the images
        :
        '''
        # find first (int) pixel relative to the center and its fractions
        fx1 = torch.abs(torch.sub(dx1, torch.floor(dx1)))
        fy1 = torch.abs(torch.sub(dy1, torch.floor(dy1)))
        if self.padding == 'valid':
            patch_middle = torch._cast_Int((self.Patch_size - 1) // 2)
            start_x1 = torch._cast_Int(patch_middle.add(torch.floor(dx1)))
            start_y1 = torch._cast_Int(patch_middle.add(torch.floor(dy1)))
            end_x1 = torch._cast_Int((W - patch_middle).add(torch.floor(dx1)))
            end_y1 = torch._cast_Int((H - patch_middle).add(torch.floor(dy1)))
        elif self.padding == 'same':
            start_x1 = torch._cast_Int(pad_L.add(torch.floor(dx1)))
            start_y1 = torch._cast_Int(pad_L.add(torch.floor(dy1)))
            end_x1 = torch._cast_Int(start_x1.add(W))
            end_y1 = torch._cast_Int(start_y1.add(H))

        # interpolate
        P1 = self.__Interp(T, [start_x1, start_y1, end_x1, end_y1], fx1, fy1)

        # find second (int) pixel relative to the center
        fx2 = torch.abs(torch.sub(dx2, torch.floor(dx2)))
        fy2 = torch.abs(torch.sub(dy2, torch.floor(dy2)))
        if self.padding == 'valid':
            patch_middle = torch._cast_Int((self.Patch_size - 1) // 2)
            start_x2 = torch._cast_Int(patch_middle.add(torch.floor(dx2)))
            start_y2 = torch._cast_Int(patch_middle.add(torch.floor(dy2)))
            end_x2 = torch._cast_Int((W - patch_middle).add(torch.floor(dx2)))
            end_y2 = torch._cast_Int((H - patch_middle).add(torch.floor(dy2)))
        elif self.padding == 'same':
            start_x2 = torch._cast_Int(pad_L.add(torch.floor(dx2)))
            start_y2 = torch._cast_Int(pad_L.add(torch.floor(dy2)))
            end_x2 = torch._cast_Int(start_x2.add(W))
            end_y2 = torch._cast_Int(start_y2.add(H))

        # interpolate
        P2 = self.__Interp(T, [start_x2, start_y2, end_x2, end_y2], fx2, fy2)

        temp = torch.sub(P1, P2)
        temp[torch.abs(temp) < 1e-5] = 0
        b = torch.sub(temp, thresh)
        ambiguity_param_pos = ambiguity_thresholds[0]
        ambiguity_param_neg = ambiguity_thresholds[1]

        # linear sigmoid function
        unclipped_B = torch.div(torch.add(b, (-1)*ambiguity_param_neg),
                                (-1)*ambiguity_param_neg + ambiguity_param_pos + 10e-30) # we add 10e-30 to
        B = torch.clamp(unclipped_B, 0 , 1)
        # B is bounded between [0,1], b is the real value of the bit function
        return B, b

    def __Interp(self, T, points_coordinates, fx, fy):
        '''
        get a bi-linear interpolation for a input tensor
        :param T: a 4D tensor (number_of_images, y_dim, x_dim)
        :param points_coordinates: 4 uint32 parameters representing the center of the pixel
            and the fractional values (x_start, y_start, x_end, y_end)
        :param fx: fraction_x float
        :param fy: fraction_y float
        :return:
        '''
        # top left fraction - T[:,y_start:y_end,x_start:x_end]
        # top right fraction - T[:,y_start:y_end,x_start+1:x_end+1]
        # bottom left fraction - T[:,y_start+1:y_end+1,x_start:x_end]
        # bottom right fraction - T[:,y_start+1:y_end+1,x_start+1:x_end+1]

        x_start, y_start, x_end, y_end = points_coordinates
        t_l_frac = (1-fx)*(1-fy)
        t_r_frac = (fx) * (1 - fy)
        b_l_frac = (1 - fx) * (fy)
        b_r_frac = (fx) * (fy)

        # output = (torch.mul(T[:,x_start:(x_end),y_start:(y_end)], t_l_frac) +
        #           torch.mul(T[:,x_start+1:(x_end) + 1,y_start:(y_end)], t_r_frac) +
        #           torch.mul(T[:,x_start:(x_end),y_start+1:(y_end) + 1], b_l_frac) +
        #           torch.mul(T[:,x_start+1:(x_end) + 1,y_start+1:(y_end) + 1], b_r_frac)
        #           )
        output = (torch.mul(T[:,y_start:(y_end),x_start:(x_end)], t_l_frac) +
                  torch.mul(T[:,y_start+1:(y_end) + 1,x_start:(x_end)], t_r_frac) +
                  torch.mul(T[:,y_start:(y_end),x_start+1:(x_end) + 1], b_l_frac) +
                  torch.mul(T[:,y_start+1:(y_end) + 1,x_start+1:(x_end) + 1], b_r_frac)
                  )
        return output


    def __initialize_ferns_parameters(self, num_of_ferns, K, Patch_Size):
        '''
        initialize_ferns_parameters randomizes values for each of the ferns' parameters
        :param num_of_ferns: number of ferns in the layer
        :param K: number of bit functions for each fern
        :param L: the patch size (L_h = L_w)
        :return: offset_vals: a num_of_ferns*K matrix containing the values of the offsets/thresholds
        '''
        if Patch_Size == 0:
            thresholds_vals = np.random.rand(num_of_ferns, K)
            return thresholds_vals
        else:
            center_of_patch = (Patch_Size - 1) / 2
            offset_vals = np.random.rand(num_of_ferns, K) * (Patch_Size - 1) - center_of_patch
            round_offset_vals = np.round(offset_vals,2)
            return round_offset_vals

    def limit_offsets_size(self):
        '''
        limits the offsets of the fern with respect to the patch size of the layer
        :param word_calc_layer: pointer to the layer holding the offsets and the patch size
        '''
        max_offset_allowed = (self.Patch_size - 1)// 2
        self.dx1.data = self.dx1.data.clamp(-max_offset_allowed, max_offset_allowed)
        self.dx2.data = self.dx2.data.clamp(-max_offset_allowed, max_offset_allowed)
        self.dy1.data = self.dy1.data.clamp(-max_offset_allowed, max_offset_allowed)
        self.dy2.data = self.dy2.data.clamp(-max_offset_allowed, max_offset_allowed)

class FernSparseTable(nn.Module):
    def __init__(self, K, num_of_ferns, num_of_active_words, D_out, input_shape, load_weights = False, args = None):
        '''
        Sparse voting layer - This operation will receive the bit functions' values and will output a new representation of the input
        :param K: number of bit functions in each fern
        :param num_of_ferns: the number of ferns that was used in the word_calc layer
        :param num_of_active_words: the number of allowed splits - this will determine how many active words will be selected
        :param D_out: the dimension of the output map
        :param input_shape
        :param load_weights: path to layer's parameters. If False, tables' weights will be initialized randomly
        '''
        super(FernSparseTable, self).__init__()
        self.num_of_ferns = num_of_ferns
        self.num_of_active_words = num_of_active_words
        self.LP = int(np.log2(num_of_active_words))
        self.num_of_bit_functions = K
        self.K_pow_2 = np.power(2, K)
        self.d_out = D_out
        self.prune_type = args.prune_type
        if load_weights is None:
            self.weights = nn.Parameter(torch.rand((num_of_ferns, 2**K, D_out)).cuda())
            # temp = torch.zeros((num_of_ferns, 2**K, D_out)).cuda()
            # temp[0, 1, 1] = 1
            # temp[0, 3, 0] = 1
            # self.weights = nn.Parameter(temp)
        else:
            weights = np.array(pd.read_csv(os.path.join(load_weights, 's2d.csv'), header = None), dtype = np.single)
            weights = np.transpose(weights)
            weights = np.reshape(weights, (num_of_ferns, 2**K, D_out))
            self.weights = nn.Parameter(torch.from_numpy(weights).cuda())

        self.bias = nn.Parameter(torch.zeros(self.d_out).cuda())
        # constant parameters that we only need to create once
        self.constant_inds = torch.from_numpy(
            self.create_constant_inds(input_shape[0], input_shape[1], input_shape[2])).permute(1,0).cuda()
        self.tensor_bit_pattern = self.create_truth_table(input_shape[0], input_shape[1], input_shape[2])
        self.args = args
        self.indices_help_tensor = torch.stack([torch.arange(K-1,-1,-1)]*input_shape[0]).unsqueeze(0).repeat(input_shape[1],1,1).permute(2,0,1).unsqueeze(0).repeat(input_shape[2],1,1,1).cuda()
        self.histogram = torch.eye(self.K_pow_2).unsqueeze(0).repeat(self.num_of_ferns, 1, 1).cuda()
        # The next two commented rows are for debug purposes
        # mat = torch.arange(0., 2**K).repeat(D_out).reshape(D_out, 2**K).transpose(1,0).cuda()
        # self.weights = mat.repeat([num_of_ferns, 1, 1]).cuda()

    def forward(self, B):
        '''
        This function start with (1) Produce the most probable word for all (images, locations),
        (2) gather the ambiguous bits and their indices from tensor T for each (image, location),
        (3) creating the indices tensor (IT) and actiovation tensor (AT)
        (4) sparse multiplication of IT, AT with the corresponding tables (holding the features of the layer)

        :param B: a 4D tensor of size (N, M*K, H, W) containing the bit function values
        :return: output: a 4D tensor (N, D_out, H, W) containing the features of the current layer. D_out is should be
        pre-determind.
        '''

        N = B.size()[0]
        mk = B.size()[1]
        H = B.size()[2]
        W = B.size()[3]

        # self.constant_inds = torch.from_numpy(
        #     self.create_constant_inds(H, W, N)).permute(1,0).cuda()

        activations = torch.zeros([N, self.num_of_active_words*self.num_of_ferns, H, W]).cuda()
        IT = torch.zeros([N, self.num_of_active_words*self.num_of_ferns, H, W]).cuda()

        # Get indices and activations for most probable words
        for m in range(self.num_of_ferns):
            start_ind = (m * self.num_of_bit_functions)
            end_ind = (m + 1) * self.num_of_bit_functions
            i, a = self.get_activations_and_indices(B[:, start_ind: end_ind, :, :])
            activations[:, m*self.num_of_active_words: (m+1)*self.num_of_active_words, :, :] = a
            IT[:, m * self.num_of_active_words: (m + 1) * self.num_of_active_words, :, :] = i

        AT = activations
        IT = torch._cast_Int(IT)
        output = torch.zeros([N * H * W, self.d_out]).cuda()

        # print average number of active words
        # if self.args.debug and H == 16:
        #     num_of_non_active_words = (AT[0,:,:,:] == 0).sum()
        #     num_of_active_words = self.num_of_ferns * self.num_of_active_words * H * W - num_of_non_active_words
        #     print('The average number of active words is {}'.format(num_of_active_words.detach().cpu().numpy()/(H*W*self.num_of_ferns)))

        inds_vector = torch.arange(0, N * H * W, dtype=torch.int32)
        rows = inds_vector.repeat(self.num_of_active_words).cuda()
        for m in range(self.num_of_ferns):
            start_ind = m * self.num_of_active_words
            end_ind = (m+1) * self.num_of_active_words
            IT_for_fern = IT[:, start_ind : end_ind, :, :]
            AT_for_fern = AT[:, start_ind : end_ind, :, :]
            cols = torch.flatten(IT_for_fern.permute(1,0,2,3))
            inds = torch.cat([rows.unsqueeze(0),cols.unsqueeze(0)], dim = 0)
            vals = torch.flatten(AT_for_fern.permute(1,0,2,3))

            Votes = torch.sparse_coo_tensor(inds, vals, [N*H*W, self.K_pow_2])
            sparse_vote = torch.sparse.mm(Votes, self.weights[m, :, :])
            output = torch.add(output, sparse_vote)
        output = output + self.bias
        output = output.permute(1,0)
        reshaped_output = output.reshape([self.d_out, N, H, W])
        output = reshaped_output.permute(1,0,2,3)

        if self.args.debug:
            return IT, AT, B, output
        return output

    def get_activations_and_indices(self, T):
        '''
        Calculate the activation and indices of each active word based on the bit function of a single fern
        :param T: a 4D tensor of size (N, K, H, W) where N is number of images, K is the number
        of bit functions, and H, W are the height and width
        :return: IT,AT of size (N, P, H, W)  - containing the indices (IT) of active words at each (image, location) and the activations (AT) of them.
        '''
        num_of_images = T.size()[0]
        num_of_bit_functions = T.size()[1]
        H = T.size()[2]
        W = T.size()[3]

        # TB = torch._cast_Int(torch.round(T))
        TB = torch._cast_Int(torch.where(T >= 0.5, torch.ones_like(T), torch.zeros_like(T)))
        WB = torch.zeros([num_of_images, H, W], dtype=torch.int32).cuda()

        bit_split_probs_tensor = torch.ones([num_of_images, H, W], dtype=torch.int32).cuda()
        T_copy = T.clone()
        # this is a bug - don't multiply by 1-T all the time
        T_copy = torch.where(T_copy < 0.5, 1-T_copy, T_copy)

        for k in range(self.num_of_bit_functions-1,-1,-1):
            TB_slice = TB[:, k, :, :]
            WB_ls = WB.__lshift__(1)

            WB = torch.add(WB_ls, TB_slice)

            bit_split_probs_tensor = torch.mul(bit_split_probs_tensor, T_copy[:,k,:,:])

        bit_split_probs_tensor = bit_split_probs_tensor.unsqueeze(1).repeat([1, self.num_of_active_words, 1, 1])

        WB_expanded = WB.unsqueeze(1)
        IT = WB_expanded.repeat(1, self.num_of_active_words, 1, 1)

        # Creating temporary tensors ABI (Ambiguous bit indices) and ABA (Ambiguous bit activations) containing
        # the most ambiguous bits from every (image, location)

        a = []
        m = []

        ABI = torch.zeros(num_of_images, self.LP, H, W).cuda()
        ABA = torch.zeros(num_of_images, self.LP, H, W).cuda()

        # insert a new prune type!
        # regular scatter - put a value of 0 or 1
        # to get the first value of the min/max value -> torch.where(a == a.min())[0][0]
        help_mat = torch.zeros_like(T).cuda()
        if self.prune_type == 1:
            BA = T.clone()
            for j in range(self.LP):
                boolean_tensor = ((BA != 0) & (BA != 1)).type(torch.float32)
                boolean_tensor[help_mat == 1] = -1
                boolean_tensor = boolean_tensor*self.indices_help_tensor
                a.append(torch.max(boolean_tensor, dim=1).indices)
                gather_inds = self.constant_inds.clone()
                gather_inds[:, 1] = a[j].flatten()
                ABI[:, j, :, :] = a[j]
                m.append(self.gather_nd(BA, gather_inds))
                gather_inds = torch._cast_Long(gather_inds)
                BA[gather_inds[:,0], gather_inds[:,1], gather_inds[:,2], gather_inds[:,3]] = 0
                help_mat[gather_inds[:,0], gather_inds[:,1], gather_inds[:,2], gather_inds[:,3]] = 1
                m[j] = torch.reshape(m[j], [num_of_images, H, W])
                ABA[:, j, :, :] = m[j]

        elif self.prune_type == 2:
            BA = torch.abs(torch.sub(T, 0.5))
            for j in range(self.LP):
                a.append(torch.argmin(BA, dim=1))
                gather_inds = self.constant_inds.clone()
                gather_inds[:,1] = a[j].flatten()
                ABI[:,j,:,:] = a[j]
                m.append(self.gather_nd(T, gather_inds))
                gather_inds = torch._cast_Long(gather_inds)
                BA[gather_inds[:,0], gather_inds[:,1], gather_inds[:,2], gather_inds[:,3]] += 1

                m[j] = torch.reshape(m[j], [num_of_images, H, W])
                ABA[:, j, :, :] = m[j]


        # Create the output tensor AT containing the word activations for all (images, locations)
        AT = torch.ones(num_of_images, self.num_of_active_words, H, W).cuda()

        for j in range(self.LP):
            ABA_slice = ABA[:, j, :, :]
            ABA_slice = ABA_slice.unsqueeze(1)
            ABA_slice_repeat = ABA_slice.repeat([1, self.num_of_active_words, 1, 1])
            bit_K = torch.where(self.tensor_bit_pattern[j], ABA_slice_repeat, 1 - ABA_slice_repeat)
            AT = torch.mul(AT, bit_K)
            # divide the total multiplication by the current bit to split
            bit_K_no_zeros = ABA_slice_repeat.clone()
            bit_K_no_zeros = torch.where(bit_K_no_zeros < 0.5, 1 - bit_K_no_zeros, bit_K_no_zeros)
            bit_split_probs_tensor = torch.div(bit_split_probs_tensor, bit_K_no_zeros)

        AT = torch.mul(AT, bit_split_probs_tensor)


        ones_matrix = torch.ones([num_of_images, H, W], dtype=torch.int32).cuda()

        # Create the output tensor IT containing the indices of active words for all (images, locations)
        for j in range(self.LP):
            ABI_slice = torch._cast_Int((ABI[:, j, :, :].cuda()))

            ABI_slice_temp = ABI_slice

            ######## notice - need to change the order of the bit!!!! ########
            # ABI_slice_temp = self.num_of_bit_functions - ABI_slice - 1
            bit_on_mask = ones_matrix.__lshift__(ABI_slice_temp)
            bit_on_mask = bit_on_mask.unsqueeze(1).repeat([1, self.num_of_active_words, 1, 1])

            bit_on_words = IT | bit_on_mask
            bit_off_mask = bit_on_mask.__xor__(torch.tensor([(2**self.num_of_bit_functions)-1]).cuda())
            bit_off_words = IT & bit_off_mask

            IT = torch.where(self.tensor_bit_pattern[j], bit_on_words, bit_off_words)

        return IT, AT

    def gather_nd(self, T, indices):
        '''
        This function gathers values from T according to indices. This function uses torch.take instead of gather_nd since
        pytorch don't use gather_nd yet.
        :param T: a 4D tensor of size (N, C, H, W) where N is the number of images,
            C is the number of channels, and H,W are the height and width
            ** gather_nd will also work for 'n_dim' tensor.
        :param indices: a Nxdim indices torch tensor holding the indices to gather.
        :return: a torch tensor holding the gathered values.
        '''
        T_size = list(T.size())
        assert len(T_size) == indices.size(1)

        indices[indices < 0] = 0

        for idx, ps in enumerate(T_size):
            indices[indices[:, idx] >= ps] = 0

        indices = indices.t().long()
        ndim = indices.size(0)
        idx = torch.zeros_like(indices[0]).long().cuda()
        m = 1

        for i in range(ndim)[::-1]:
            idx += indices[i] * m
            m *= T.size(i)

        return torch.take(T, idx)

    def create_constant_inds(self, H, W, num_of_images):
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

    def create_truth_table(self, H, W, N):
        # Create ‘truth table’ binary mask tensors
        tensor_bit_pattern = []
        const_2_tensor = torch.tensor(2, dtype=torch.int32).cuda()
        for j in range(self.LP):
            replicate = const_2_tensor.pow(j)

            zeros_and_ones = torch.cat([torch.zeros(replicate, dtype=torch.int32).cuda(), torch.ones(replicate, dtype=torch.int32).cuda()])
            bit_pattern = zeros_and_ones.repeat(int(self.num_of_active_words / (2 * replicate)))
            bit_pattern_repeat = bit_pattern.repeat([N, H, W, 1])
            tensor_bit_pattern.append(bit_pattern_repeat.permute(0,3,1,2).bool())

        return tensor_bit_pattern

class FernBitWord_tabular(nn.Module):

    def __init__(self, num_of_ferns, K, d_in, args, device):
        '''
        Constructor for the Fern bit word encoder for tabular data.
        This layer includes the following learnable parameters:
            - alpha - the given weight for each of the features
            - th - the threshold for each of the bit function in each fern
        the length of each of these parameter lists is hence number_of_ferns*number_of_bit_functions

        :param num_of_ferns: number of ferns to used in the layer
        :param K: number of bit functions in each fern
        :param d_in: the number of features in the input tensor
        '''
        super(FernBitWord_tabular, self).__init__()
        self.num_of_ferns = num_of_ferns
        self.num_of_bit_functions = K
        self.alpha = nn.Parameter((torch.from_numpy(np.random.rand(num_of_ferns, K, d_in)).type(torch.float32)).to(device))
        self.th = nn.Parameter(torch.from_numpy(np.zeros([num_of_ferns, K])).to(device))
        self.ambiguity_thresholds = init_ambiguity_thresholds(self.num_of_ferns, self.num_of_bit_functions, device)
        self.anneal_state_params = init_anneal_state_tabular(args)
        self.args = args
        self.device = device
        self.alpha_after_softmax = np.zeros(self.alpha.shape) # torch.zeros_like(self.alpha)
        #this part is for debug only!
        # temp = np.zeros([num_of_ferns, K, d_in])
        # temp[0, 0, 4] = np.log(3)
        # temp[0, 1, 4] = np.log(0.5)
        # temp[0, 2, 4] = np.log(1)
        # self.alpha = nn.Parameter(torch.from_numpy(temp).cuda())
        # self.th = nn.Parameter(torch.from_numpy(np.zeros([num_of_ferns, K])).cuda())
        # self.ambiguity_thresholds = init_ambiguity_thresholds(self.num_of_ferns, self.num_of_bit_functions)
        # self.ambiguity_thresholds[0][0, :] = 6
        # self.ambiguity_thresholds[0][1, :] = -6
        # self.tempature = 1
        # self.anneal_state_params = init_anneal_state()

    def forward(self, T):
        '''
        applied on the input tensor, this layer slices the tensor according to the number of bit function and the number of ferns
        and calculates the bit function (for now only comparison between 2 pixels and a threshold is implemented) for each of the
        ferns.
        :param T: a 2D tensor of size (N, D_in) where N is the number of examples,
            D is the the number of features
        :return: output: a 3D tensor of size (N, M, K) where M is the number_of_ferns, K is the number_of_bit_functions.
            This tensor hold the bit function values for each cell in the input.
        '''
        Bits = torch.zeros([T.size(0), self.num_of_ferns, self.num_of_bit_functions]).to(self.device)
        bit_functions_values = []
        for m in range(self.num_of_ferns):
            current_alpha = self.alpha[m]
            current_alpha_with_tempature = torch.mul(current_alpha, self.anneal_state_params['tempature'])
            softmax_alpha = torch.nn.functional.softmax(current_alpha_with_tempature, dim=1)
            current_th = self.th[m]
            Bits[:, m, :], bit_values = self.__Bit(T, softmax_alpha, current_th, self.ambiguity_thresholds[m])
            bit_functions_values.append(bit_values)
            self.alpha_after_softmax[m] = softmax_alpha.detach().cpu().numpy()   #Aharon: this line creates the problem  - the mysterious bug. Now fixed
        if self.anneal_state_params['count_till_update'] == self.anneal_state_params['batch_till_update']:
            # save the bit function values for the annealing mechanism
            self.bit_functions_values = bit_functions_values   # This also may be problematic: Detach it.

        return Bits

    def __Bit(self, T, alpha, thresh, ambiguity_thresholds):
        '''
        compute a complete ferns with K bit functions for all examples
        :param T: a matrix of size (N, D_in) where N is the number of examples,
            and D_in is the number of features
        :param alpha: a matric of size (num_of_bit_functions, D_in) containing the weights for each cell of each of the bit functions in a given fern
        :param thresh: a matric of size (num_of_bit_functions, D_in) containing the threshold of bit function
        :param ambiguity_thresholds: vector of size (K,2) holding the ambiguity thresholds (pos, neg)
        :return: B: a matrix of size (N, K) containing the bit function value (in the range [0,1]) for each exmaple
        :return: b: a 3D tensor of size (N, K) containing the bit function value before bounding it, for each example
        '''
        temp = torch.mm(T, torch.transpose(alpha,1,0))
        temp[torch.abs(temp) < 1e-5] = 0
        b = torch.sub(temp, thresh)
        ambiguity_param_pos = ambiguity_thresholds[0]
        ambiguity_param_neg = ambiguity_thresholds[1]

        # linear sigmoid function
        unclipped_B = torch.div(torch.add(b, (-1)*ambiguity_param_neg),
                                (-1)*ambiguity_param_neg + ambiguity_param_pos + 10e-30) # we add 10e-30 to
        B = torch.clamp(unclipped_B, 0 , 1)
        # B is bounded between [0,1], b is the real value of the bit function
        return B, b

    # def __initialize_ferns_parameters(self, num_of_ferns, K, d_in):
    #     '''
    #     initialize_ferns_parameters randomizes values for each of the ferns' parameters
    #     :param num_of_ferns: number of ferns in the layer
    #     :param K: number of bit functions for each fern
    #     :param d_in: the number of features in the input
    #     :return: weights: a num_of_ferns*K matrix containing the values of the offsets/thresholds
    #     '''
    #     weights = np.random.rand(num_of_ferns, K, d_in)
    #     return weights

class FernSparseTable_tabular(nn.Module):
    def __init__(self, K, num_of_ferns, num_of_active_words, D_out, input_shape, prune_type, device, args = None):
        '''
        Sparse voting layer - This operation will receive the bit functions' values and will output a new representation of the input
        :param K: number of bit functions in each fern
        :param num_of_ferns: the number of ferns that was used in the word_calc layer
        :param num_of_active_words: the number of allowed splits - this will determine how many active words will be selected
        :param D_out: the dimension of the output map
        :param input_shape
        :param load_weights: path to layer's parameters. If False, tables' weights will be initialized randomly
        '''
        super(FernSparseTable_tabular, self).__init__()
        self.num_of_ferns = num_of_ferns
        self.num_of_active_words = num_of_active_words
        self.LP = int(np.log2(num_of_active_words))
        self.num_of_bit_functions = K
        self.K_pow_2 = np.power(2, K)
        self.d_out = D_out
        self.prune_type = prune_type
        self.device = device
        self.weights = nn.Parameter(torch.rand((num_of_ferns, 2**K, D_out)).to(self.device))
        self.bias = nn.Parameter(torch.zeros(self.d_out).to(self.device))
        # constant parameters that we only need to create once
        self.constant_inds = torch.from_numpy(self.create_constant_inds(input_shape[0])).to(self.device)
        self.tensor_bit_pattern = self.create_truth_table(input_shape[0])
        self.args = args
        self.indices_help_tensor = torch.arange(K-1,-1,-1).unsqueeze(0).repeat(input_shape[0], 1).to(self.device)
        # The next two commented rows are for debug purposes
        # mat = torch.arange(0., 2**K).repeat(D_out).reshape(D_out, 2**K).transpose(1,0).cuda()
        # self.weights = mat.repeat([num_of_ferns, 1, 1]).cuda()

    def forward(self, B):
        '''
        This function start with (1) Produce the most probable word for all (images, locations),
        (2) gather the ambiguous bits and their indices from tensor T for each (image, location),
        (3) creating the indices tensor (IT) and actiovation tensor (AT)
        (4) sparse multiplication of IT, AT with the corresponding tables (holding the features of the layer)

        :param B: a 3D tensor of size (N, M, K) containing the bit function values
        :return: output: a 2D tensor (N, D_out) containing the features of the current layer. D_out is should be
        pre-determind.
        '''
        N = B.size()[0]
        M = B.size()[1]
        K = B.size()[2]

        activations = torch.zeros([N, self.num_of_active_words*self.num_of_ferns]).to(self.device)
        IT = torch.zeros([N, self.num_of_active_words*self.num_of_ferns]).to(self.device)

        # Get indices and activations for most probable words
        for m in range(self.num_of_ferns):
            current_fern = B[:, m, :]
            if (current_fern - current_fern.int()).sum() == 0:
                activity = torch.zeros([N, self.num_of_active_words])
                activity[:,0] = 1
                words = torch.zeros([N, self.num_of_active_words])
                words[:, 0] = torch.sum(torch.mul(current_fern, torch.pow(2, self.indices_help_tensor)), 1)
            else:
                words, activity = self.get_activations_and_indices(current_fern)
            activations[:, m*self.num_of_active_words: (m+1)*self.num_of_active_words] = activity
            IT[:, m * self.num_of_active_words: (m + 1) * self.num_of_active_words] = words

        AT = activations
        IT = torch._cast_Int(IT)
        output = torch.zeros([N, self.d_out]).to(self.device)
        # debug - check the number of average active words
        AT_bin = AT > 0.0001
        #print('average number of active words is %i' %((AT_bin.sum()//self.args.batch_size)//M))

        inds_vector = torch.arange(0, N, dtype=torch.int32)
        rows = inds_vector.repeat(self.num_of_active_words).to(self.device)
        for m in range(self.num_of_ferns):
            start_ind = m * self.num_of_active_words
            end_ind = (m+1) * self.num_of_active_words
            IT_for_fern = IT[:, start_ind : end_ind]
            AT_for_fern = AT[:, start_ind : end_ind]
            cols = torch.flatten(IT_for_fern.permute(1,0))
            inds = torch.cat([rows.unsqueeze(0),cols.unsqueeze(0)], dim = 0)
            vals = torch.flatten(AT_for_fern.permute(1,0))

            Votes = torch.sparse_coo_tensor(inds, vals, [N, self.K_pow_2]).to(self.device)
            sparse_vote = torch.sparse.mm(Votes, self.weights[m])
            output = torch.add(output, sparse_vote)
        output = output + self.bias
        # reshaped_output = output.reshape([self.d_out, N])
        # output = reshaped_output.permute(1,0)

        return output

    def get_activations_and_indices(self, T):
        '''
        Calculate the activation and indices of each active word based on the bit function of a single fern
        :param T: a 4D tensor of size (N, K, H, W) where N is number of images, K is the number
        of bit functions, and H, W are the height and width
        :return: IT,AT of size (N, P, H, W)  - containing the indices (IT) of active words at each (image, location) and the activations (AT) of them.
        '''
        num_of_examples = T.size()[0]
        num_of_bit_functions = T.size()[1]

        if num_of_examples < self.indices_help_tensor.shape[0]:
            indices_help_tensor = torch.arange(num_of_bit_functions-1,-1,-1).unsqueeze(0).repeat(num_of_examples, 1).to(self.device)
            constant_inds = torch.from_numpy(self.create_constant_inds(num_of_examples)).to(self.device)
            tensor_bit_pattern = self.create_truth_table(num_of_examples)
        else:
            indices_help_tensor = self.indices_help_tensor
            constant_inds = self.constant_inds
            tensor_bit_pattern = self.tensor_bit_pattern

        # TB = torch._cast_Int(torch.round(T))
        TB = torch._cast_Int(torch.where(T >= 0.5, torch.ones_like(T), torch.zeros_like(T)))
        WB = torch.zeros(num_of_examples, dtype=torch.int32).to(self.device)

        bit_split_probs_tensor = torch.ones(num_of_examples, dtype=torch.int32).to(self.device)
        T_copy = T.clone()
        # this is a bug - don't multiply by 1-T all the time
        T_copy = torch.where(T_copy < 0.5, 1-T_copy, T_copy)

        for k in range(self.num_of_bit_functions-1,-1,-1):
            TB_slice = TB[:, k]
            WB_ls = WB.__lshift__(1)

            WB = torch.add(WB_ls, TB_slice)

            bit_split_probs_tensor = torch.mul(bit_split_probs_tensor, T_copy[:,k])

        bit_split_probs_tensor = bit_split_probs_tensor.unsqueeze(1).repeat([1, self.num_of_active_words])

        WB_expanded = WB.unsqueeze(1)
        IT = WB_expanded.repeat(1, self.num_of_active_words)

        # Creating temporary tensors ABI (Ambiguous bit indices) and ABA (Ambiguous bit activations) containing
        # the most ambiguous bits from every (image, location)

        a = []
        m = []

        ABI = torch.zeros(num_of_examples, self.LP).to(self.device)
        ABA = torch.zeros(num_of_examples, self.LP).to(self.device)

        help_mat = torch.zeros_like(T).to(self.device)
        if self.prune_type == 1:
            BA = T.clone()
            for j in range(self.LP):
                boolean_tensor = ((BA != 0) & (BA != 1)).type(torch.float32)
                boolean_tensor[help_mat == 1] = -1
                boolean_tensor = boolean_tensor*indices_help_tensor
                a.append(torch.max(boolean_tensor, dim=1).indices)
                ABI[:, j] = a[j]
                gather_inds = constant_inds.clone()
                gather_inds[:, 1] = a[j].flatten()
                gather_inds = torch._cast_Long(gather_inds)
                m.append(BA[gather_inds[:,0], gather_inds[:,1]])
                BA[gather_inds[:,0], gather_inds[:,1]] = 0
                help_mat[gather_inds[:,0], gather_inds[:,1]] = 1
                ABA[:, j] = m[j]

        elif self.prune_type == 2:
            BA = torch.abs(torch.sub(T, 0.5))
            for j in range(self.LP):
                a.append(torch.argmin(BA, dim=1))
                gather_inds = self.constant_inds.clone()
                gather_inds[:,1] = a[j].flatten()
                ABI[:,j] = a[j]
                gather_inds = torch._cast_Long(gather_inds)
                m.append(self.gather(T, gather_inds))
                BA[gather_inds[:,0], gather_inds[:,1]] += 1

                m[j] = torch.reshape(m[j], num_of_examples)
                ABA[:, j] = m[j]

        # Create the output tensor AT containing the word activations for all (images, locations)
        AT = torch.ones(num_of_examples, self.num_of_active_words).to(self.device)

        for j in range(self.LP):
            ABA_slice = ABA[:, j]
            ABA_slice = ABA_slice.unsqueeze(1)
            ABA_slice_repeat = ABA_slice.repeat([1, self.num_of_active_words])
            bit_K = torch.where(tensor_bit_pattern[j], ABA_slice_repeat, 1 - ABA_slice_repeat)
            AT = torch.mul(AT, bit_K)
            # divide the total multiplication by the current bit to split
            bit_K_no_zeros = ABA_slice_repeat.clone()
            bit_K_no_zeros = torch.where(bit_K_no_zeros < 0.5, 1 - bit_K_no_zeros, bit_K_no_zeros)
            bit_split_probs_tensor = torch.div(bit_split_probs_tensor, bit_K_no_zeros)

        AT = torch.mul(AT, bit_split_probs_tensor)


        ones_matrix = torch.ones(num_of_examples, dtype=torch.int32).to(self.device)

        # Create the output tensor IT containing the indices of active words for all (images, locations)
        for j in range(self.LP):
            ABI_slice = torch._cast_Int((ABI[:, j].to(self.device)))
            ABI_slice_temp = ABI_slice

            bit_on_mask = ones_matrix.__lshift__(ABI_slice_temp)
            bit_on_mask = bit_on_mask.unsqueeze(1).repeat([1, self.num_of_active_words])

            bit_on_words = IT | bit_on_mask
            bit_off_mask = bit_on_mask.__xor__(torch.tensor([(2**self.num_of_bit_functions)-1]).to(self.device))
            bit_off_words = IT & bit_off_mask

            IT = torch.where(tensor_bit_pattern[j], bit_on_words, bit_off_words)

        return IT, AT

    def create_constant_inds(self, num_of_exampels):
        '''
        :param num_of_images:
        :return:
        '''
        first_col = np.arange(0, num_of_exampels)
        ones_vector = np.ones(num_of_exampels)
        constant_inds = np.vstack((first_col, ones_vector)).transpose()
        return constant_inds

    def create_truth_table(self, N):
        # Create ‘truth table’ binary mask tensors
        tensor_bit_pattern = []
        const_2_tensor = torch.tensor(2, dtype=torch.int32).to(self.device)
        for j in range(self.LP):
            replicate = const_2_tensor.pow(j)

            zeros_and_ones = torch.cat([torch.zeros(replicate, dtype=torch.int32).to(self.device), torch.ones(replicate, dtype=torch.int32).to(self.device)])
            bit_pattern = zeros_and_ones.repeat(int(self.num_of_active_words / (2 * replicate)))
            bit_pattern_repeat = bit_pattern.repeat([N, 1])
            tensor_bit_pattern.append(bit_pattern_repeat.bool().to(self.device))

        return tensor_bit_pattern
