import tensorflow as tf
import torch
import numpy as np

def gather_nd(params, indices):
    param_size = list(params.size())
    assert len(param_size) == indices.size(1)

    indices[indices<0] = 0

    for idx, ps in enumerate(param_size):
        indices[indices[:,idx] >= ps] = 0

    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)

    return torch.take(params, idx)

idx = np.array([
    [0,0,0,0],
    [0,0,1,0],
    [0,1,1,0],
    [2,1,1,0],
    [1,1,4,0]
])

mtx = np.reshape(range(120),(2,3,4,5))
print(mtx)

tf_result = tf.gather_nd(mtx,idx)

with tf.Session() as sess:
    print('TF', sess.run(tf_result))

torch_mtx = torch.from_numpy(mtx)
torch_idx = torch.from_numpy(idx)

torch_result = gather_nd(torch_mtx, torch_idx)

print('torch', torch_result)