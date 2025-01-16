import torch
import numpy as np
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        coords = np.array([coo.row, coo.col])
        i = torch.LongTensor(coords)
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape, dtype=torch.float)


    