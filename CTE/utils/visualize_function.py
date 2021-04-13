from Sandboxes.Farjon.CTE.utils.help_funcs import to_numpy
import matplotlib.pyplot as plt
import torch
import math
import os
import numpy as np


def visualize_network_parameters(args, rho, fern_params1, fern_params2, patch_center, loss, fig=None, ax=None, lines=None):
    '''

    :param args: dictionary containing the following:
        - batch - a number representing the current batch
        - epoch - a number representing the current epoch
        - loss - the current loss
        - index - the x-axis index for the graphs
    :param rho:
    :param fern_params1:
    :param fern_params2:
    :param patch_cetner:
    :param loss:
    :param fig:
    :param ax:
    :param lines:
    :return:
    '''
    #  Initializing figure
    if (args['epoch'] == 0 and args['batch'] == 0):

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15),
                           sharex=False, sharey=False)
        fig.suptitle('Batch number ' + str(args['batch']) + ' of epoch ' + str(args['epoch']))

        # Plot sample of  train data
        lines ={'loss': ax[0, 0].plot(args['index'], to_numpy(loss), color='b', label='loss value', marker = '+')}
        ax[0, 0].set_title('Loss of current batch')
        ax[0, 0].legend(loc="upper right")
        ax[0,0].set_xlim(-1, 20000)
        ax[0,0].set_ylim(-1, 2000)

        lines['Rho'] = ax[0,1].plot(args['index'], rho, color='b', label='Rho value', marker = '+')
        ax[0, 1].set_title('Current Rho value')
        ax[0, 1].legend(loc="upper right")
        ax[0, 1].set_xlim(-1, 20000)
        ax[0, 1].set_ylim(0, 1)

        x1 = fern_params1[0].item() + patch_center
        y1 = fern_params1[1].item() + patch_center
        x2 = fern_params1[2].item() + patch_center
        y2 = fern_params1[3].item() + patch_center
        lines['bit_function_pixel_1'] = ax[1,0].plot(x1, y1, color='b', label='P1', marker = '+')
        lines['bit_function_pixel_2'] = ax[1,0].plot(x2, y2, color='g', label='P2', marker = '+')
        ax[1, 0].set_title('a bit function of first fern')
        ax[1, 0].legend(loc="upper right")
        ax[1, 0].set_xlim(0, patch_center*2 + 1)
        ax[1, 0].set_ylim(0, patch_center*2 + 1)

        x1 = fern_params2[0].item() + patch_center
        y1 = fern_params2[1].item() + patch_center
        x2 = fern_params2[2].item() + patch_center
        y2 = fern_params2[3].item() + patch_center
        lines['bit_function_pixel_1'] = ax[1,1].plot(x1, y1, color='b', label='P1', marker = '+')
        lines['bit_function_pixel_2'] = ax[1,1].plot(x2, y2, color='g', label='P2', marker = '+')
        ax[1, 1].set_title('a bit function of first fern')
        ax[1, 1].legend(loc="upper right")
        ax[1, 1].set_xlim(0, patch_center*2 + 1)
        ax[1, 1].set_ylim(0, patch_center*2 + 1)
    else:
        fig.suptitle('Batch number ' + str(args['batch']) + ' of epoch ' + str(args['epoch']))

        lines ={'loss': ax[0, 0].plot(args['index'], to_numpy(loss), color='b', label='loss value', marker = '+')}
        ax[0, 0].set_title('Loss of current batch')
        ax[0,0].set_xlim(-1, 100000)
        ax[0,0].set_ylim(-1, 500)

        lines['Rho'] = ax[0,1].plot(args['index'], rho, color='b', marker = '+')
        ax[0, 1].set_title('Current Rho value')
        ax[0, 1].set_xlim(-1, 100000)
        ax[0, 1].set_ylim(0, 1)

        x1 = fern_params1[0].item() + patch_center
        y1 = fern_params1[1].item() + patch_center
        x2 = fern_params1[2].item() + patch_center
        y2 = fern_params1[3].item() + patch_center
        lines['bit_function_pixel_1'] = ax[1,0].plot(x1, y1, color='b', marker = '+')
        lines['bit_function_pixel_2'] = ax[1,0].plot(x2, y2, color='g', marker = '+')
        ax[1, 0].set_title('First bit function of first fern')
        ax[1, 0].set_xlim(0, patch_center*2 + 1)
        ax[1, 0].set_ylim(0, patch_center*2 + 1)

        x1 = fern_params2[0].item() + patch_center
        y1 = fern_params2[1].item() + patch_center
        x2 = fern_params2[2].item() + patch_center
        y2 = fern_params2[3].item() + patch_center
        lines['bit_function_pixel_1'] = ax[1,1].plot(x1, y1, color='b', marker = '+')
        lines['bit_function_pixel_2'] = ax[1,1].plot(x2, y2, color='g', marker = '+')
        ax[1, 1].set_title('First bit function of first fern')
        ax[1, 1].set_xlim(0, patch_center*2 + 1)
        ax[1, 1].set_ylim(0, patch_center*2 + 1)

    # plt.pause(.00001)
    plt.savefig(os.path.join(args['graph_path'], 'statistics_graph_stage_' +  str(args['stage']) + '.jpg'))

    return fig, ax, lines


