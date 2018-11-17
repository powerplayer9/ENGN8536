import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import customDataExtractor
import customDataLoader
import accuracyCheck
import LossCheck
import Train
import Net


# Parameters
batch_size = 10
valid_size = 1000
Epoch = 2

# Extracting & obtaining all image paths & labels from zipfile
trainPath, trainLabel, testPath, testLabel = customDataExtractor.extractor('Dog-data.zip')

# Loading data from path & obtaining labels for training & testing data
trainData = customDataLoader.loader(trainPath,trainLabel)
testData = customDataLoader.loader(testPath,testLabel)

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
net = Net.Net()
#net.to(device)

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
xAxis = np.arange(1, Epoch + 1 )


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
