import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    # The CNN layers in order
    '''
    5x5 Convolutional Layer with 32 filters, stride 1 and padding 2.        -> 32 x 32 x 3 -> 32 x 32 x 32
    ReLU Activation Layer                                                   -> 32 x 32 x 32 -> 32 x 32 x 32
    Batch Normalization Layer                                               -> 32 x 32 x 32 -> 32 x 32 x 32
    2x2 Max Pooling Layer with a stride of 2                                -> 32 x 32 x 32 -> 16 x 16 x 32
    3x3 Convolutional Layer with 64 filters, stride 1 and padding 1.        -> 16 x 16 x 32 -> 16 x 16 x 64
    ReLU Activation Layer                                                   -> 16 x 16 x 64 -> 16 x 16 x 64
    Batch Normalization Layer                                               -> 16 x 16 x 64 -> 16 x 16 x 64
    2x2 Max Pooling Layer with a stride of 2                                -> 16 x 16 x 64 -> 8 x 8 x 64
    Fully-conneted layer with 1024 output units                             -> 8 x 8 x 64 -> 1024
    ReLU Activation Layer                                                   -> 1024 -> 1024
    Drop-out Layer                                                          -> 1024 -> 1024
    Fully-connected layer with 10 output units                              -> 1024 -> 10
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.drop = nn.Dropout2d(p = 0.3)

    def forward(self, x):
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))

        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = drop(x)
        x = self.fc2(x)
        return x

