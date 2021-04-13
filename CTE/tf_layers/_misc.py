import keras
import keras.backend as K
import tensorflow as tf
import numpy as np



class FernBitWord(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        fern_parameters = args[0]
        self.num_of_ferns = fern_parameters['num_of_ferns']
        self.num_of_bit_functions = fern_parameters['K']
        self.dx1_init = fern_parameters['dx1']
        self.dx2_init = fern_parameters['dx2']
        self.dy1_init = fern_parameters['dy1']
        self.dy2_init = fern_parameters['dy2']
        self.th_init = fern_parameters['thresholds']

        super(FernBitWord, self).__init__(**kwargs)

    def build(self, input_shape):
        '''

        :param input_shape:
        :return:
        '''
        self.dx1 = tf.get_variable("dx1", initializer=self.dx1_init)
        self.dx2 = tf.get_variable("dx2", initializer=self.dx2_init)
        self.dy1 = tf.get_variable("dy1", initializer=self.dy1_init)
        self.dy2 = tf.get_variable("dy2", initializer=self.dy2_init)
        self.th = tf.get_variable("th", initializer=self.th_init)
        self.trainable_weights=[self.dx1, self.dx2, self.dy1, self.dy2, self.th]

    def call(self, inputs, **kwargs):
        '''
        applied on the input tensor, this layer slices the tensor according to the number of bit function and the number of ferns
        :param inputs: a 4D tensor of size (N, D, H, W) where N is the number of images,
            D is the the number of channels, and H,W are the height and width
        :param kwargs:
        :return:
        '''
        T = inputs
        # take parameters from input
        # channels_to_slice = kwargs.values()
        dx1 = K.get_value(self.dx1)
        dx2 = K.get_value(self.dx2)
        dy1 = K.get_value(self.dy1)
        dy2 = K.get_value(self.dy2)
        thresholds = K.get_value(self.th)

        # local parameters
        dx1_max = K.max(K.max(K.abs(dx1)))
        dx2_max = K.max(K.max(K.abs(dx2)))
        dy1_max = K.max(K.max(K.abs(dy1)))
        dy2_max = K.max(K.max(K.abs(dy2)))
        L = K.max([dx1_max, dx2_max, dy1_max, dy2_max])
        # L = tf.cast(10, tf.float64)

        Bits = []

        for m in range(self.num_of_ferns):
            for k in range(self.num_of_bit_functions):
                # current_channel = channels_to_slice[m, k]
                current_channel = 1
                channel = T[:, current_channel, :, :]

                # to use spatial_2d_padding - padding should be a tuple of two tuples of tf.int32 values
                pad_L = tf.cast(tf.ceil(L), tf.int32)
                padding = ((pad_L,pad_L), (pad_L,pad_L))
                channel_padded = tf.keras.backend.spatial_2d_padding(channel, padding)
                bit_answer = self.__Bit(channel_padded, dx1[m, k], dx2[m, k], dy1[m, k], dy2[m, k], thresholds[m, k], L)
                Bits.append(bit_answer)

        output = K.concatenate(Bits, axis=1)

        return output

    def __Bit(self, T, dx1, dx2, dy1, dy2, t, L):
        '''
        compute a single bit function for all images
        :param T: a 4D tensor of size (N, 1, H, W) where N is the number of images,
            1 is the sliced channel, and H,W are the height and width
        :param dx1: the first pixel offset x-dim
        :param dy1: the first pixel offset y-dim
        :param dx2: the second pixel offset x-dim
        :param dy2: the second pixel offset y-dim
        :param t: the threshold of the linear sigmoid function
        :param L: the patch size (LxL)
        :return:
        '''

        # take W,H values of the network's input
        H = tf.shape(T)[1]
        W = tf.shape(T)[2]

        # find first (int) pixel relative to the center and its fractions
        x1 = tf.cast(tf.add(L ,tf.floor(dx1)), tf.int32)
        y1 = tf.cast(tf.add(L ,tf.floor(dy1)), tf.int32)
        fx1 = tf.subtract(dx1, tf.floor(dx1))
        fy1 = tf.subtract(dy1, tf.floor(dy1))

        # interpolate
        P1 = self.__Interp(T, [x1, y1, W, H], fx1, fy1)

        # find second (int) pixel relative to the center
        x2 = tf.cast(tf.add(L ,tf.floor(dx2)), tf.int32)
        y2 = tf.cast(tf.add(L ,tf.floor(dy2)), tf.int32)
        fx2 = tf.subtract(dx2, tf.floor(dx2))
        fy2 = tf.subtract(dy2, tf.floor(dy2))

        # interpolate
        P2 = self.__Interp(T, [x2, y2, W, H], fx2, fy2)

        # TODO - probably will need to expand t to the size of P_i
        b = tf.subtract(tf.subtract(P1, P2), t)
        ambiguity_param = 2.0
        # linear sigmoid function
        un_clipped_B = tf.divide(tf.add(b, 0.5), tf.multiply(ambiguity_param,2))
        B = tf.clip_by_value(un_clipped_B, 0, 1)

        return B

    def __Interp(self, T, points_coordinates, fx, fy):
        '''
        get a bi-linear interpolation for a input tensor
        :param T: a 4D tensor (number_of_images, 1, y_dim, x_dim)
        :param points_coordinates: 4 uint32 parameters representing the center of the pixel
            and the fractional values (x_c, y_c, x_p, y_p)
        :param fx: fraction_x float
        :param fy: fraction_y float
        :return:
        '''
        x_c, y_c, x_p, y_p = points_coordinates
        t_l_frac = tf.cast((1 - fx) * (1 - fy), tf.float32)
        t_r_frac = tf.cast((fx) * (1 - fy), tf.float32)
        b_l_frac = tf.cast((1 - fx) * (fy), tf.float32)
        b_r_frac = tf.cast((fx) * (fy), tf.float32)

        # top left fraction
        t_l_slice = tf.slice(T, [0, 0, y_c, x_c], [-1, -1, y_p - 1, x_p - 1])
        t_l = tf.multiply(t_l_slice, t_l_frac)
        # top right fraction
        t_r_slice = tf.slice(T, [0, 0, y_c, x_c + 1], [-1, -1, y_p - 1, x_p])
        t_r = tf.multiply(t_r_slice, t_r_frac)
        # bottom left fraction
        b_l_slice = tf.slice(T, [0, 0, y_c + 1, x_c], [-1, -1, y_p, x_p - 1])
        b_l = tf.multiply(b_l_slice, b_l_frac)
        # bottom right fraction
        b_r_slice = tf.slice(T, [0, 0, y_c + 1, x_c + 1], [-1, -1, y_p, x_p])
        b_r = tf.multiply(b_r_slice, b_r_frac)

        output = tf.add(tf.add(tf.add(t_l, t_r), b_l), b_r)
        return output


class FernSparseTable(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        fern_parameters = args[0]
        constant_inds = args[1]
        self.constant_inds = tf.get_variable("constant_values",initializer=constant_inds)
        self.num_of_ferns = fern_parameters['num_of_ferns']
        self.num_of_active_words = fern_parameters['num_of_active_words']
        self.num_of_bit_functions = fern_parameters['K']
        # self.d_out = d_out

        super(FernSparseTable, self).__init__(**kwargs)

    def call(self, inputs):
        B = inputs
        N = tf.shape(B)[0]
        mk = tf.shape(B)[1]
        H = tf.shape(B)[2]
        W = tf.shape(B)[3]

        # num_of_bit_functions = tf.cast((mk/self.num_of_ferns), tf.int32)

        activations = []
        indices = []
        for m in range(self.num_of_ferns):
            start_ind = (m * self.num_of_bit_functions) + 1
            end_ind = (m + 1) * self.num_of_bit_functions
            a, i = self.__get_activations_and_indices(B[:, start_ind : end_ind, :, :])
            activations.append(a)
            indices.append(i)

        AT = K.concatenate(activations, axis = 1)
        IT = K.concatenate(indices, axis = 1)

        output = K.zeros([N*H*W*self.num_of_active_words, self.d_out])

        for m in range(self.num_of_ferns):
            rows = tf.cast(K.flatten(K.repeat(tf.range(N*H*W), [1, self.num_of_active_words])), tf.uint64)
            cols = tf.cast(K.flatten(K.permute_dimensions(IT, [1, 0, 2, 3])), tf.uint64)

            inds = [rows, cols]
            vals = K.flatten(K.permute_dimensions(AT, [1 ,0 ,2 ,3]))

            Votes = tf.SparseTensor(inds, vals, [N*H*W*self.num_of_active_words, tf.pow(2,self.num_of_bit_functions)])
            # TODO - lets talk about the weights and how to initialize them
            output = tf.add(output, tf.sparse.sparse_dense_matmul(Votes, ))

        output = K.permute_dimensions(K.repeat(output, [self.d_out, N, H, W]), [1, 0, 2, 3])
        return output


    def __get_activations_and_indices(self, T):
        num_of_images = tf.shape(T)[0]
        num_of_bit_functions = tf.shape(T)[1]
        H = tf.shape(T)[2]
        W = tf.shape(T)[3]

        TB = tf.cast(K.round(T), tf.uint32)
        WB = tf.zeros([num_of_images,H,W], dtype=tf.uint32)

        for k in range(self.num_of_bit_functions):
            WB = tf.bitwise.bitwise_or(tf.bitwise.left_shift(WB, tf.ones([1],dtype=tf.uint32)),TB[:,k,:,:])

        # TODO - check the repeat function
        # this will be the output tensor
        WB_expanded = tf.expand_dims(WB, 1)
        IT = K.tile(WB_expanded, [1, self.num_of_active_words, 1, 1])

        a = []
        m = []
        LP = int(round(np.log2(self.num_of_active_words)))
        BA = tf.abs(tf.subtract(T, 0.5))

        for j in range(LP):
            a.append(tf.cast(K.argmin(BA, axis=1), tf.float64))
            gather_inds = tf.Variable(self.constant_inds.initialized_value())
            # possible bug here
            tf.assign(gather_inds [:,1], a[j])
            a[j] = K.expand_dims(a[j], 1)
            gather_inds = tf.cast(gather_inds, tf.int32)
            m.append(tf.gather_nd(T, gather_inds))
            BA = tf.scatter_nd_add(BA, gather_inds, K.ones(num_of_images * H * W, 1))
            m[j] = K.reshape(m[j], [num_of_images, 1, H, W])

        ABI = K.concatenate(a, 1)
        ABA = K.concatenate(m, 1)

        tensor_bit_pattern = []
        for j in range(LP):
            replicate = 2**j
            bit_pattern = K.repeat([np.zeros(replicate,1), np.ones(replicate,1)], self.num_of_active_words/(2*replicate))
            tensor_bit_pattern.append(K.repeat(bit_pattern, [num_of_images, 1, H, W]))

        AT = K.ones(num_of_images, self.num_of_active_words)

        for j in range(LP):
            ABA_repeat = K.repeat(ABA[:, j, :, :], [1, self.num_of_active_words, 1, 1])
            bit_K = keras.backend.switch(tensor_bit_pattern[j], ABA_repeat, 1 - ABA_repeat)
            AT = tf.multiply(AT, bit_K)


        ones_matrix = K.ones([num_of_images, 1, H, W])

        for j in range(LP):
            bit_on_mask = tf.bitwise.left_shift(ones_matrix, ABI[:,j,:,:])
            bit_off_mask = tf.bitwise.bitwise_xor(bit_on_mask, K.ones([num_of_images, 1, H, W]))
            bit_on_words = tf.bitwise.bitwise_or(IT,K.repeat(bit_on_mask, [1, self.num_of_active_words, H, W]))
            bit_off_words = tf.bitwise.bitwise_or(IT, K.repeat(bit_off_mask, [1, self.num_of_active_words, H, W]))
            IT = K.switch(tensor_bit_pattern[j], bit_on_words, bit_off_words)

        return IT, AT


