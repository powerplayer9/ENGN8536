import torch.nn as nn
import torch.nn.functional as F
import MaxOut


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(self.norm1(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetMax(nn.Module):

    def __init__(self):
        super(NetMax, self).__init__()
        self.conv1 = nn.Conv2d(3, 32*4, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64*4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.maxOut1 = MaxOut.MaxOut()

    def forward(self, x):
        x = self.maxOut1(self.conv1(x),4)
        x = self.pool(self.norm1(x))
        x = self.maxOut1(self.conv2(x),4)
        x = self.pool(self.norm2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class NetMaxDrop(nn.Module):

    def __init__(self):
        super(NetMaxDrop, self).__init__()
        self.conv1 = nn.Conv2d(3, 32*4, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64*4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.maxOut1 = MaxOut.MaxOut()
        self.drop = nn.Dropout2d(p=0.3)

    def forward(self, x):
        x = self.maxOut1(self.conv1(x),4)
        x = self.pool(self.norm1(x))
        x = self.drop(x)
        x = self.maxOut1(self.conv2(x),4)
        x = self.pool(self.norm2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

