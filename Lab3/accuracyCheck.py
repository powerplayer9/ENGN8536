import torch


def accuracyCheck(dataInput, net1, string):
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net1.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataInput:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net1(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total * 100
    print('Accuracy of the network on the %s images: %d %%' % (string, acc))

    return acc