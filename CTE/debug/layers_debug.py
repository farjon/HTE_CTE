import numpy as np
import torch.nn as nn
import torch
from annealing_mechanism_functions import init_ambiguity_thresholds
from annealing_mechanism_functions import init_anneal_state

# pytorch uses NxCxHxW dimension order!!

class FernBitWord(nn.Module):

    def __init__(self, num_of_ferns, K, Patch_Size):#, BC_params):
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
        '''
        super(FernBitWord, self).__init__()
        self.num_of_ferns = num_of_ferns
        self.num_of_bit_functions = K

        # The following commeneted parameters are for debug purposes only
        self.dx1 = nn.Parameter(torch.tensor([[0, 0, 0, 0.]]).cuda())
        self.dy1 = nn.Parameter(torch.tensor([[0, 0, 0, 0.]]).cuda())
        self.dx2 = nn.Parameter(torch.tensor([[-5., 5, 0, 0]]).cuda())
        self.dy2 = nn.Parameter(torch.tensor([[0., 0, -5, 5]]).cuda())
        self.th = nn.Parameter(torch.tensor([[0.,0.,0., 0]]).cuda())

        # self.dx1 = nn.Parameter(torch.tensor([BC_params[0]]).cuda())
        # self.dy1 = nn.Parameter(torch.tensor([BC_params[1]]).cuda())
        # self.dx2 = nn.Parameter(torch.tensor([BC_params[2]]).cuda())
        # self.dy2 = nn.Parameter(torch.tensor([BC_params[3]]).cuda())
        # self.th = nn.Parameter(torch.tensor([BC_params[4]]).cuda())


        # TODO - change into aragne cycle
        self.channels_to_slice = torch.from_numpy(np.random.random([num_of_ferns, K])).cuda()

        self.ambiguity_thresholds = init_ambiguity_thresholds(self.num_of_ferns, self.num_of_bit_functions)
        self.anneal_state_params = init_anneal_state()

    def forward(self, x):
        '''
        applied on the input tensor, this layer slices the tensor according to the number of bit function and the number of ferns
        and calculates the bit function (for now only comparison between 2 pixels and a threshold is implemented) for each of the
        ferns.
        :param x: a 4D tensor of size (N, D, H, W) where N is the number of images,
            D is the the number of channels, and H,W are the height and width
        :return: output: a 4D tensor of size (N, M*K, H, W) where M*K is the number_of_ferns*number_of_bit_functions.
            This tensor hold the bit function values for each pixels location in each of the images.
        '''
        T = x

        # take parameters from input
        number_of_channels = T.size()[1] - 1
        channels_to_slice = torch.mul(number_of_channels, self.channels_to_slice)
        T_size = T.size()
        # local parameters
        dx1_max = torch.max(torch.max(torch.abs(self.dx1)))
        dx2_max = torch.max(torch.max(torch.abs(self.dx2)))
        dy1_max = torch.max(torch.max(torch.abs(self.dy1)))
        dy2_max = torch.max(torch.max(torch.abs(self.dy2)))
        # self.dx1.retain_grad()
        # self.dx1.register_hook(lambda x: print(x))

        L = torch.max(torch.tensor([dx1_max, dx2_max, dy1_max, dy2_max]).cuda())

        Bits = torch.zeros((T_size[0], self.num_of_bit_functions*self.num_of_ferns, T_size[2], T_size[3])).cuda()

        bit_functions_values = []
        index = 0
        for m in range(self.num_of_ferns):
            fern_bit_function_values = torch.zeros(self.num_of_bit_functions, T_size[0]*T_size[2]*T_size[3]).cuda()
            for k in range(self.num_of_bit_functions):
                current_channel = int(channels_to_slice[m, k].item())
                channel = T[:, current_channel, :, :]

                # take W,H values of the network's input before padding
                H = T.size()[2]
                W = T.size()[3]

                pad_L = torch.ceil(L + 1)
                padding = nn.ConstantPad2d(torch._cast_Int(pad_L).item(), 0)
                channel_padded = padding(channel)
                Bits[:, index, :, :], bit_values = self.__Bit(channel_padded, self.dx1[m, k], self.dx2[m, k], self.dy1[m, k], self.dy2[m, k], self.th[m, k], pad_L, H, W, self.ambiguity_thresholds[m][:,k])
                # print(Bits.grad)
                fern_bit_function_values[k, :] = bit_values.view(-1)
                index = index + 1
            bit_functions_values.append(fern_bit_function_values)
        output = Bits
        # save the bit function values for the annealing mechanism
        self.bit_functions_values = bit_functions_values

        return output

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
        :param ambiguity_thresholds: vector of size (2,1) holding the ambiguity thresholds (pos, neg)
        :return: B: a 3D tensor of size (N, H, W) containing the bit function value (in the range [0,1]) for pixel location in each of the images
        :return: b: a 3D tensor of size (N, H, W) containing the bit function value before bounding it, for pixel location in each of the images
        :
        '''
        # find first (int) pixel relative to the center and its fractions
        fx1 = torch.abs(torch.sub(dx1, torch.floor(dx1)))
        fy1 = torch.abs(torch.sub(dy1, torch.floor(dy1)))
        start_x1 = torch._cast_Int(pad_L.add(torch.floor(dx1)))
        start_y1 = torch._cast_Int(pad_L.add(torch.floor(dy1)))
        end_x1 = torch._cast_Int(start_x1.add(W))
        end_y1 = torch._cast_Int(start_y1.add(H))

        # interpolate
        P1 = self.__Interp(T, [start_x1, start_y1, end_x1, end_y1], fx1, fy1)

        # find second (int) pixel relative to the center
        fx2 = torch.abs(torch.sub(dx2, torch.floor(dx2)))
        fy2 = torch.abs(torch.sub(dy2, torch.floor(dy2)))
        start_x2 = torch._cast_Int(pad_L.add(torch.floor(dx2)))
        start_y2 = torch._cast_Int(pad_L.add(torch.floor(dy2)))
        end_x2 = torch._cast_Int(start_x2.add(W))
        end_y2 = torch._cast_Int(start_y2.add(H))

        # interpolate
        P2 = self.__Interp(T, [start_x2, start_y2, end_x2, end_y2], fx2, fy2)

        temp = torch.sub(P1, P2)
        b = torch.sub(temp, thresh)
        ambiguity_param_pos = ambiguity_thresholds[0]
        ambiguity_param_neg = ambiguity_thresholds[1]

        # linear sigmoid function
        unclipped_B = torch.div(torch.add(b, (-1)*ambiguity_param_neg), (-1)*ambiguity_param_neg + ambiguity_param_pos)
        # B = torch.clamp(unclipped_B, 0, 1)
        clipped_up_B = torch.where(unclipped_B > 20, torch.ones_like(unclipped_B).cuda(), unclipped_B)
        B = torch.where(clipped_up_B < -20, torch.zeros_like(unclipped_B).cuda(), clipped_up_B)
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
        x_start, y_start, x_end, y_end = points_coordinates
        t_l_frac = (1-fx)*(1-fy)
        t_r_frac = (fx) * (1 - fy)
        b_l_frac = (1 - fx) * (fy)
        b_r_frac = (fx) * (fy)

        # top left fraction
        t_l_slice = T[:,y_start:y_end,x_start:x_end]
        t_l = torch.mul(t_l_slice, t_l_frac)
        # top right fraction
        t_r_slice = T[:,y_start:y_end,x_start+1:x_end+1]
        t_r = torch.mul(t_r_slice, t_r_frac)
        # bottom left fraction
        b_l_slice = T[:,y_start+1:y_end+1,x_start:x_end]
        b_l = torch.mul(b_l_slice, b_l_frac)
        # bottom right fraction
        b_r_slice = T[:,y_start+1:y_end+1,x_start+1:x_end+1]
        b_r = torch.mul(b_r_slice, b_r_frac)

        output = t_l + t_r + b_l + b_r
        return output


    def __initialize_ferns_parameters(self, num_of_ferns, K, Patch_Size):
        '''
        initialize_ferns_parameters randomizes values for each of the ferns' parameters
        :param num_of_ferns: number of ferns in the layer
        :param K: number of bit functions for each fern
        :param L: the patch size (L_h = L_w)
        :return:
        '''
        if Patch_Size == 0:
            thresholds_vals = np.random.rand(num_of_ferns, K)
            return thresholds_vals
        else:
            center_of_patch = (Patch_Size - 1) / 2
            offset_vals = np.random.rand(num_of_ferns, K) * (Patch_Size - 1) - center_of_patch
            round_offset_vals = np.round(offset_vals,2)
            return round_offset_vals


class FernSparseTable(nn.Module):
    def __init__(self, constant_inds, K, num_of_ferns, num_of_active_words, D_out):
        '''
        Sparse voting layer -
        :param constant_inds:
        :param K:
        :param num_of_ferns:
        :param num_of_active_words:
        '''
        super(FernSparseTable, self).__init__()
        self.constant_inds = torch.from_numpy(constant_inds).permute(1,0).cuda()
        self.num_of_ferns = num_of_ferns
        self.num_of_active_words = num_of_active_words
        self.num_of_bit_functions = K
        self.d_out = D_out
        # self.weights = nn.Parameter(torch.rand((num_of_ferns, 2**K, D_out)).cuda())
        # The next two commented rows are for debug purposes
        mat = torch.arange(0., 2**K).repeat(D_out).reshape(D_out, 2**K).transpose(1,0).cuda()
        self.weights = nn.Parameter(mat.repeat([num_of_ferns, 1, 1])).cuda()

    def forward(self, x):
        '''
        This function start with (1) Produce the most probable word for all (images, locations),
        (2) gather the ambiguous bits and their indices from tensor T for each (image, location),
        (3) creating the indices tensor (IT) and actiovation tensor (AT)
        (4) sparse multiplication of IT, AT with the corresponding tables (holding the features of the layer)

        :param x: a 4D tensor of size (N, M*K, H, W) containing the bit function values
        :return: output: a 4D tensor (N, D_out, H, W) containing the features of the current layer. D_out is should be
        pre-determind.
        '''

        B = x

        N = B.size()[0]
        mk = B.size()[1]
        H = B.size()[2]
        W = B.size()[3]

        activations = torch.zeros([N, self.num_of_active_words*self.num_of_ferns, H, W]).cuda()
        indices = torch.zeros([N, self.num_of_active_words*self.num_of_ferns, H, W]).cuda()

        # Get indices and activations for most probable words
        for m in range(self.num_of_ferns):
            start_ind = (m * self.num_of_bit_functions)
            end_ind = (m + 1) * self.num_of_bit_functions
            i, a = self.get_activations_and_indices(B[:, start_ind: end_ind, :, :])
            activations[:, m*self.num_of_active_words: (m+1)*self.num_of_active_words, :, :] = a
            indices[:, m * self.num_of_active_words: (m + 1) * self.num_of_active_words, :, :] = i

        AT = activations
        IT = torch._cast_Long(indices)
        output = torch.zeros([N * H * W, self.d_out]).cuda()

        for m in range(self.num_of_ferns):
            inds_vector = torch.arange(0,N * H * W)
            rows = inds_vector.repeat(self.num_of_active_words).cuda()
            start_ind = m * self.num_of_active_words
            end_ind = (m+1) * self.num_of_active_words
            IT_for_fern = IT[:, start_ind : end_ind, :, :]
            AT_for_fern = AT[:, start_ind : end_ind, :, :]
            cols = torch.flatten(IT_for_fern.permute(1,0,2,3))
            inds = torch.cat([rows.unsqueeze(0),cols.unsqueeze(0)], dim = 0)
            vals = torch.flatten(AT_for_fern.permute(1,0,2,3))

            Votes = torch.sparse_coo_tensor(inds, vals, [N*H*W, 2**self.num_of_bit_functions])
            sparse_vote = torch.sparse.mm(Votes, self.weights[m, :, :])
            output = torch.add(output, sparse_vote)
        output = output.permute(1,0)
        reshaped_output = output.reshape([self.d_out, N, H, W])
        output = reshaped_output.permute(1,0,2,3)
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

        TB = torch._cast_Int(torch.round(T))
        WB = torch.zeros([num_of_images, H, W], dtype=torch.int32).cuda()

        for k in range(self.num_of_bit_functions):
            TB_slice = TB[:, k, :, :]
            WB_ls = WB.__lshift__(1)

            WB = torch.add(WB_ls, TB_slice)

        WB_expanded = WB.unsqueeze(1)
        IT = WB_expanded.repeat(1, self.num_of_active_words, 1, 1)

        # Creating temporary tensors ABI (Ambiguous bit indices) and ABA (Ambiguous bit activations) containing
        # the most ambiguous bits from every (image, location)

        a = []
        m = []
        num_of_active_words_tensor = torch._cast_Float(torch.tensor(self.num_of_active_words))
        LP = torch._cast_Int(torch.round(torch.log2(num_of_active_words_tensor))).item()
        BA = torch.abs(torch.sub(T, 0.5))

        ABI = torch.zeros(num_of_images, LP, H, W).cuda()
        ABA = torch.zeros(num_of_images, LP, H, W).cuda()

        for j in range(LP):
            a.append(torch.argmin(BA, dim=1))
            gather_inds = self.constant_inds.clone()
            gather_inds[:,1] = a[j].flatten()
            ABI[:,j,:,:] = a[j]
            m.append(self.gather_nd(T, gather_inds))
            gather_inds = torch._cast_Long(gather_inds)
            BA[gather_inds[:,0], gather_inds[:,1], gather_inds[:,2], gather_inds[:,3]] += 1

            m[j] = torch.reshape(m[j], [num_of_images, H, W])
            ABA[:, j, :, :] = m[j]


        # Create ‘truth table’ binary mask tensors
        tensor_bit_pattern = []
        const_2_tensor = torch.tensor(2, dtype=torch.int32)
        for j in range(LP):
            replicate = const_2_tensor.pow(j).item()

            zeros_and_ones = torch.cat([torch.zeros(replicate, dtype=torch.int32), torch.ones(replicate, dtype=torch.int32)])
            bit_pattern = zeros_and_ones.repeat(int(self.num_of_active_words / (2 * replicate)))
            bit_pattern_repeat = bit_pattern.repeat([num_of_images, H, W, 1])
            tensor_bit_pattern.append(bit_pattern_repeat.permute(0,3,1,2).bool().cuda())

        # Create the output tensor AT containing the word activations for all (images, locations)
        AT = torch.ones(num_of_images, self.num_of_active_words, H, W).cuda()

        for j in range(LP):
            ABA_slice = ABA[:, j, :, :]
            ABA_slice = ABA_slice.unsqueeze(0).permute(1,0,2,3)
            ABA_slice_repeat = ABA_slice.repeat([1, self.num_of_active_words, 1, 1])
            bit_K = torch.where(tensor_bit_pattern[j], ABA_slice_repeat, 1 - ABA_slice_repeat)
            AT = torch.mul(AT, bit_K)

        ones_matrix = torch.ones([num_of_images, H, W], dtype=torch.int32).cuda()

        # Create the output tensor IT containing the indices of active words for all (images, locations)
        for j in range(LP):
            ABI_slice = torch._cast_Int((ABI[:, j, :, :].cuda()))
            ABI_slice_temp = self.num_of_bit_functions - ABI_slice - 1
            bit_on_mask = ones_matrix.__lshift__(ABI_slice_temp)
            bit_on_mask = bit_on_mask.unsqueeze(0).permute(1,0,2,3).repeat([1, self.num_of_active_words, 1, 1])

            bit_on_words = IT | bit_on_mask
            bit_off_mask = bit_on_mask.__xor__(torch.tensor([(2**self.num_of_bit_functions)-1]).cuda())
            bit_off_words = IT & bit_off_mask

            IT = torch.where(tensor_bit_pattern[j], bit_on_words, bit_off_words)

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


