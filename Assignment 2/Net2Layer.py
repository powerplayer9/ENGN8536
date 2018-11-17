import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    '''
    Fully-conneted layer with x output units                             -> 784 -> x
    sigmoid Activation Layer                                             -> x -> x
    Fully-connected layer with 10 output units                           -> x -> 10
    softmax Activation Layer                                             -> 10 -> 10
    '''

    def __init__(self, hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

