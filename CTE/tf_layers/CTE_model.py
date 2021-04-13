import sys
import numpy as np
import Farjon.CTE.layers as layers
import torch


def create_constant_inds(H, W, num_of_images):
    '''

    :param data_input_size: N*H*W*C tensor holding the sizes of the input to the network
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

def CTE_Model(args):

    constant_inds = create_constant_inds(args.input_size)

    input = Input(shape=args.input_size, batch_shape=(args.batch_size, args.input_size[0], args.input_size[1], args.input_size[2]))
    x = layers.FernBitWord()(input)
    x = layers.FernSparseTable( constant_inds)(x)
    pred = Dense(args.num_of_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=pred)
    model.compile(optimizer=args.optimizer, loss=args.loss)

    return model






