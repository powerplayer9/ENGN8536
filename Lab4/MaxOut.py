import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxOut(nn.Module):
    # def __init__(self, kValue):
    #     self.k = nn.Parameter(kValue)

    def forward(self, data, kValue):
        self.k = kValue
        dataDepth = data.shape[1]
        finalDepth = int(dataDepth / self.k)
        newshapeDim = (data.shape[0], finalDepth, self.k, data.shape[2], data.shape[3])
        data = data.view(newshapeDim)
        # print('data before max=', np.shape(data))
        output, i = torch.max(data[:, :, :], 2)
        # print('data after max=', np.shape(output))
        return output
