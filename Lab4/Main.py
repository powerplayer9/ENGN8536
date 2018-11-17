import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import checkFile
import accuracyCheck
import LossCheck
import Train
import Net


if __name__ == '__main__':

    # # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    # Parameters
    batch_size = 100
    valid_size = 1000
    Epoch = 1

    # Loading datasets & class labels
    trainloader, validloader, testloader, classes = checkFile.initFiles(batch_size, valid_size)


    net = Net.Net()
    # net.to(device)

    # Set up cross-entropy loss
    # criterion = nn.CrossEntropyLoss()

    # Set up Adam optimizer, with 1e-3 learning rate and betas=(0.9, 0.999)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

    # Initialize Accuracy List
    trainAccuracy = np.zeros(1)
    validAccuracy = np.zeros(1)

    # Initialize Loss List
    trainLoss = np.zeros(1)
    validLoss = np.zeros(1)

    # Train your model
    for epoch in range(Epoch):  # loop over the dataset multiple times

        trainLossTemp, net = Train.train(trainloader, net, 'Training', epoch)
        trainLoss = np.append(trainLoss, trainLossTemp)
        validLoss = np.append(validLoss, LossCheck.lossCheck(validloader, net, 'Validation', epoch))

        trainAccuracy = np.append(trainAccuracy, accuracyCheck.accuracyCheck(trainloader, net, 'Training'))
        validAccuracy = np.append(validAccuracy, accuracyCheck.accuracyCheck(validloader, net, 'Validation'))

    print('Fin...')

    # Deleting 1st index value
    trainLoss = np.delete(trainLoss,0)
    validLoss = np.delete(validLoss,0)
    trainAccuracy = np.delete(trainAccuracy,0)
    validAccuracy = np.delete(validAccuracy,0)

    # Saving the network
    torch.save(net, './Save')

    # Plots

    # Making X axis
    xAxis = np.arange(1, Epoch)

    # Training loss vs. epochs
    fig1 = plt.figure()
    plt.plot(xAxis, trainLoss)
    fig1.suptitle('Training Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=16)

    # Training accuracy vs. epochs
    fig2 = plt.figure()
    plt.plot(xAxis, trainAccuracy)
    fig2.suptitle('Training Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)

    # Validation loss vs epochs
    fig3 = plt.figure()
    plt.plot(xAxis, validLoss)
    fig3.suptitle('Validation Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=16)

    # Validation accuracy vs. epochs
    fig4 = plt.figure()
    plt.plot(xAxis, validAccuracy)
    fig4.suptitle('Validation Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)