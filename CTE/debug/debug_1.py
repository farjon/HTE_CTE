# run 'match_examples_to_matlab_debug.py' and put a breaking point in line 326
# when the cursor gets to line 326, put another breaking point at line 58 in 'CTE_two_layers_inter_loss.py'
import torch
import numpy as np
def debug_fern_AT_IT(AT, IT, B, ambiguity_thresholds):
    x = 1
    y = 0
    F = 0

    # Ambiguity thrsholds
    t = ambiguity_thresholds[0].detach().cpu().numpy()

    # bit functions
    # b = res(1).aux{F}(:, x + 16 * (y - 1))';

    # probabilities after sigmoid
    p = B[0, 10*F:10*(F+1), y, x].detach().cpu().numpy()
    # p = min(max((b - t(2,:)). / (t(1,:) - t(2,:)), 0), 1)

    # indices
    inds = IT[0, 16*F:16:(F+1), y, x].detach().cpu().numpy()
    # inds = res(2).x.indices(:, x, y, F, 1)'-1


    # activations
    a = AT[0, 16*F:16:(F+1), y, x].detach().cpu().numpy()
    # a = res(2).x.activations(:, x, y, F, 1)'

    split_inds = [(p != 0) & (p != 1)]
    split_inds = np.where(split_inds)[1]
    if len(split_inds) == 0:
        return
    # split_inds = find(p~ = 0 & p~ = 1);
    if len(split_inds) > 4:
        split_inds = split_inds[:4]
    print(split_inds)

    Words = []
    a2 = []
    for i in range(a.size):
        ind = inds[i]
        tmp = np.binary_repr(ind, 10)
        Words.append(tmp)
        tmp_P = 1
        # for j in range(9,0,-1):
        for j in range(10):
            if tmp[j] == '0':
                tmp_P = tmp_P * (1 - p[9-j])
            else:
                tmp_P = tmp_P * p[9-j]

        a2.append(tmp_P)
    a2 = np.array(a2)
    print(a)
    print(a2)
    if np.max(np.abs(a-a2)) > 1e-4:
        print('a and a2 are not the same\n')
    else:
        print('a and a2 are good !!!!!!!!!!!!!!!!!!!!\n')

    # W = double(Words(i,:)-48);
    # W = W(end:-1: 1);
    # a3(i) = prod(p. ^ W) * prod((1 - p). ^ (1 - W));
    # end
    # a2
    # a3
    #
    # % Words
    # Words
    # Words(:, 11 - split_inds(end: -1:1)) % should
    # be
    # a
    # truth
    # table

