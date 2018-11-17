import torch.nn as nn
from torchvision import models


def modifiedResnet():

    # Load a pretrained model and reset final fully connected layer.
    model_conv = models.resnet18(pretrained=True)

    # Freezing all layers except fully connected layer
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    # print(num_ftrs)
    model_conv.fc = nn.Linear(num_ftrs, 2)

    return model_conv