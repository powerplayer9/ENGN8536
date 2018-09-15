import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def initFiles(batch_size,valid_size):

    # Transforms Done
    # randomly crop 32x32 as input to the network
    # zero-pad 4 pixels on each side of the input image
    # randomly flip the image left and right
    # normalize the data to range between (-1, 1)
    transform = transforms.Compose(
        [transforms.RandomCrop(32,
                               padding=4,
                               pad_if_needed=False),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transformtest = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download/Check Training dataset
    trainData = torchvision.datasets.CIFAR10(root='./data',             # Destination
                                             train=True,                # Download training set
                                             download=True,             # Download if file not found
                                             transform=transform)       # If any transforms are to be applied
    print('TrainData Done')
    print('Size Train Data:', len(trainData))

    # Split training into train and validation
    indices = torch.randperm(len(trainData))
    train_indices = indices[:len(indices) - valid_size]         # Taining         1 - 49 000 (49 000)
    valid_indices = indices[len(indices) - valid_size:]         # Validation 49 001 - 50 000 ( 1 000)

    # print(type(trainData))

    # Loading Training & Validation Data
    # trainloader, validloader = torch.utils.data.random_split(trainData, (49000,1000))
    # print('TrainLoader Done')
    # print('Size Train:', len(trainloader))
    # print('ValidLoader Done')
    # print('Size Train:', len(validloader))

    # Loading Training Data
    trainloader = torch.utils.data.DataLoader(trainData,
                                             batch_size=batch_size,
                                             sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                                             num_workers=2)
    print('TrainLoader Done')
    print('Size Train:', len(trainloader))

    # Loading Validation Data
    validloader = torch.utils.data.DataLoader(trainData,
                                             batch_size=batch_size,
                                             sampler=torch.utils.data.SubsetRandomSampler(valid_indices),
                                             num_workers=2)
    print('ValidLoader Done')
    print('Size Train:', len(validloader))

    # Download/Check Test dataset
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transformtest)
    print('TestSet Done')
    print('Size Train:', len(testset))

    # Loading Test dataset
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)
    print('TestLoader Done')
    print('Size Train:', len(testloader))

    # Hard defining classes in the Dataset
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Returing the datasets & class labels
    return trainloader, validloader, testloader, classes