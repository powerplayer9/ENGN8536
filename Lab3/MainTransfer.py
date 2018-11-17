import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import customDataExtractor
import transferCustomDataLoader
import accuracyCheck
import LossCheck
import transferTrain
import transferResnetModel


# Parameters
batch_size = 100
valid_size = 1000
Epoch = 2

# Extracting & obtaining all image paths & labels from zipfile
trainPath, trainLabel, testPath, testLabel = customDataExtractor.extractor('Dog-data.zip')

# Loading data from path & obtaining labels for training & testing data
trainData = transferCustomDataLoader.loader(trainPath,trainLabel)
testData = transferCustomDataLoader.loader(testPath,testLabel)

# Splitting Indices from training & validation data
indices = torch.randperm(len(trainData))
train_indices = indices[:len(indices) - valid_size]  # Taining         1 - 19 000 (19 000)
valid_indices = indices[len(indices) - valid_size:]  # Validation 19 001 - 20 000 ( 1 000)

# Making miniBatches
trainloader = DataLoader(trainData, batch_size=batch_size,
                       sampler=torch.utils.data.SubsetRandomSampler(train_indices))
validloader = DataLoader(trainData, batch_size=batch_size,
                       sampler=torch.utils.data.SubsetRandomSampler(valid_indices))
testloader = DataLoader(testData, batch_size=batch_size,
                        shuffle=True)

# Defining the Net
net = transferResnetModel.modifiedResnet()
#net.to(device)

# Initialize Accuracy List
trainAccuracy = np.zeros(1)
validAccuracy = np.zeros(1)

# Initialize Loss List
trainLoss = np.zeros(1)
validLoss = np.zeros(1)

# Train your model
for epoch in range(Epoch):  # loop over the dataset multiple times

    trainLossTemp, net = transferTrain.train(trainloader, net, 'Training', epoch)
    trainLoss = np.append(trainLoss, trainLossTemp)
    validLoss = np.append(validLoss, LossCheck.lossCheck(validloader, net, 'Validation', epoch))

    trainAccuracy = np.append(trainAccuracy, accuracyCheck.accuracyCheck(trainloader, net, 'Training'))
    validAccuracy = np.append(validAccuracy, accuracyCheck.accuracyCheck(validloader, net, 'Validation'))

print('Fin...')
