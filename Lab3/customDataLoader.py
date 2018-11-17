from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class loader(Dataset):

    def __init__(self, dataPath=None, dataLabel=None):
        self.imagePath = dataPath
        self.imageLabelSet = dataLabel

        # Applying required transforms
        self.transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.imagePath)

    def __getitem__(self, idx):
        #print('\tcalling Dataset:__getitem__ for index =%d' % idx)
        imageFile = Image.open(self.imagePath[idx])
        imgaeTensor = self.transform(imageFile)
        imageLabel = self.imageLabelSet[idx]
        #imageLabel = torch.long(imageLabel)
        #print(type(imageLabel))

        return imgaeTensor, imageLabel