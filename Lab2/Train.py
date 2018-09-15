import torch
import torch.nn as nn
import torch.optim as optim


def train(dataInput, net1, string, epoch):
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net1.to(device)

    # Set up cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Set up Adam optimizer, with 1e-3 learning rate and betas=(0.9, 0.999)
    optimizer = optim.Adam(net1.parameters(), lr=1e-3, betas=(0.9, 0.999))

    for i, data in enumerate(dataInput, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        lossVal = loss.item()

    print('[%d, %5d] %s loss: %.3f' %
           (epoch + 1, i + 1, string, lossVal))

    return lossVal, net1