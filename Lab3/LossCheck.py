import torch
import torch.nn as nn


def lossCheck (dataInput, net1, string, epoch):
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net1.to(device)

    # Set up cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    for i, data in enumerate(dataInput, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward + backward + optimize
        outputs = net1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # print statistics
        lossVal = loss.item()

    print('[%d, %5d] %s loss: %.3f' %
          (epoch + 1, i + 1, string, lossVal))

    return lossVal