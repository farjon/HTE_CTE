import numpy as np
import sys
import torch
from layers._misc import *
import utils.annealing_mechanism_functions as anneal_functions
print('__Python VERSION:', sys.version)
print('_pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()
use_cuda = True

print("USE CUDA=" + str (use_cuda))
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

from CTE.layers._misc import *

def create_syn_image(H, W, N):


    image = np.zeros([N, 1, H, W])
    image[0, :, 9:19, 9:19] = 100
    image[1, :, 1:11, 1:11] = 100
    return image



def debug_interp():
    Fern_bit_word = FernBitWord(10, 8, 7)

    a = np.array([[1],[2],[3],[4],[5]])
    b = a.transpose()
    tensor_n = torch.tensor(a.dot(b))
    tensor_n = tensor_n.unsqueeze(0)

    answer = Fern_bit_word.Interp(tensor_n, [1,1,4,4], 0.25, 0)
    print(tensor_n)
    print(answer)
    return


def debug_bit_function():
    Fern_bit_word = FernBitWord(10, 8, 7)

    a = np.array([[1], [2], [3], [4], [5]])
    b = a.transpose()
    tensor_n = torch.tensor(a.dot(b)).cuda()

    dx1 = torch.tensor([1.5]).cuda()
    dy1 = torch.tensor([1.]).cuda()
    dx2 = torch.tensor([-1.5]).cuda()
    dy2 = torch.tensor([-1.0]).cuda()
    L = torch.max(torch.tensor([torch.abs(dx1),
                   torch.abs(dx2),
                   torch.abs(dy1),
                   torch.abs(dy2)])).cuda()

    # take W,H values of the network's input before padding
    H = tensor_n.size()[0]
    W = tensor_n.size()[1]

    pad_L = torch.ceil(L + 1).cuda()
    padding = nn.ConstantPad2d(torch._cast_Int(pad_L).item(), 0).cuda()
    tensor_n = padding(tensor_n).cuda()

    tensor_n = tensor_n.unsqueeze(0).cuda()

    tensor_B = Fern_bit_word.Bit(tensor_n, dx1, dx2, dy1, dy2, 5, pad_L, H, W)

    print(tensor_B)
    return

def debug_complete_fern():
    Fern_bit_word = FernBitWord(1, 4, 13)

    tensor_n = torch.from_numpy(create_syn_image(28, 28, 2)).cuda()

    output = Fern_bit_word.forward(tensor_n)
    print(output)
    return

def create_constant_inds(H, W, num_of_images):
    x_temp = np.linspace(0, W - 1, num=W, dtype='int')
    y_temp = np.linspace(0, H - 1, num=H, dtype='int')
    Xv, Yv = np.meshgrid(x_temp, y_temp)
    X = np.tile(Xv.flatten(), num_of_images)
    Y = np.tile(Yv.flatten(), num_of_images)
    first_col = np.repeat(np.arange(0, num_of_images), W * H)
    ones_vector = np.ones([W * H * num_of_images])
    constant_inds = np.vstack((first_col, ones_vector, Y, X))
    return constant_inds

def debug_gather_nd():
    W = 10
    H = 10
    num_of_images = 2
    # TODO - take constant_inds for self.
    constant_inds = create_constant_inds(H, W, num_of_images)


    a = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    b = a.transpose()
    tensor_n = torch.tensor(a.dot(b)).cuda()
    tensor_n = tensor_n.repeat(2, 2, 1, 1)

    fern_sparse_table = FernSparseTable(constant_inds, 8, 10, 8, 5)


    indices = [[0, 0, 0, 1],
               [0, 0, 1 ,2],
               [1, 1, 0, 0],
               [0, 1, 0, 1]]

    first = tensor_n[indices[0][0], indices[0][1], indices[0][2], indices[0][3]]
    second = tensor_n[indices[1][0], indices[1][1], indices[1][2], indices[1][3]]
    third = tensor_n[indices[2][0], indices[2][1], indices[2][2], indices[2][3]]
    fourth = tensor_n[indices[3][0], indices[3][1], indices[3][2], indices[3][3]]
    print(first)
    print(second)
    print(third)
    print(fourth)

    gathered_values = fern_sparse_table.gather_nd(tensor_n, torch.tensor(indices).cuda())
    print(gathered_values)
    return

def debug_get_activations_and_indices():
    num_of_bit_functions = 4
    W = 50
    H = 40
    num_of_images = 2
    num_of_ferns = 1
    Fern_bit_word = FernBitWord(num_of_ferns, num_of_bit_functions, 7)

    tensor_n = torch.from_numpy(create_syn_image(H, W, num_of_images)).cuda()

    # TODO - take constant_inds for self.
    constant_inds = np.transpose(create_constant_inds(H, W, num_of_images))
    fern_sparse_table = FernSparseTable(constant_inds, num_of_bit_functions, num_of_ferns, 4, 5)

    output = Fern_bit_word.forward(tensor_n)

    IT, AT = fern_sparse_table.get_activations_and_indices(output)
    return

def debug_complete_sparse_table():
    num_of_bit_functions = 4
    W = 28
    H = 28
    num_of_images = 2
    num_of_ferns = 1
    Fern_bit_word = FernBitWord(num_of_ferns, num_of_bit_functions, 13)

    tensor_n = torch.from_numpy(create_syn_image(H, W, num_of_images)).cuda()

    fern_sparse_table = FernSparseTable(num_of_bit_functions, num_of_ferns, 4, 5, [H - 12, W - 12, num_of_images])

    output = Fern_bit_word.forward(tensor_n)
    output = fern_sparse_table.forward(output)
    return

def debug_update_anneal_params():
    anneal_params = anneal_functions.init_anneal_state()
    ambiguity_thresh = anneal_functions.init_ambiguity_thresholds(1, 5)
    np.random.seed(10)
    # bit_values = torch.from_numpy(np.random.rand(5,1000)).cuda()
    bit_values = torch.from_numpy(np.random.uniform(-5, 5, [5, 1000])).cuda()
    bit_function_values = [bit_values]
    ambiguity_thresh = anneal_functions.update_ambiguity_thresholds(anneal_params, ambiguity_thresh, bit_function_values)

def main():
    debug_choice = 'sparse_table'

    if debug_choice == 'interp':
        debug_interp()
    elif debug_choice == 'bit_func':
        debug_bit_function()
    elif debug_choice == 'fern':
        debug_complete_fern()
    elif debug_choice == 'gather_nd':
        debug_gather_nd()
    elif debug_choice == 'get_activ_and_inds':
        debug_get_activations_and_indices()
    elif debug_choice == 'sparse_table':
        debug_complete_sparse_table()
    elif debug_choice == 'annealing':
        debug_update_anneal_params()




if __name__ == '__main__':
    torch.cuda.set_device(0)
    np.random.seed(10)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()