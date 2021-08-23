from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    CNN from Ciprijanovic et al. (2020) Astronomy and Computing, 32, 100390
    """

    def __init__(self, features_size=32*18*18):
        """
        Convolutional neural network receives images and encodes them into an array of size `features_size`.

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.
        """

        super(Encoder, self).__init__()

        self.features_size = features_size

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(8)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)
        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.batchn3(self.conv3(x))))
        x = x.view(-1, self.features_size)
        return x

class Classifier(nn.Module):
    """
    CNN from Ciprijanovic et al. (2020) Astronomy and Computing, 32, 100390
    """

    def __init__(self, features_size=32*18*18, num_classes=3):
        """
        Neural network that receives an array of size `features_size` and classifies it into `num_classes` classes.

        Arguments:
        ----------
        features_size: int
            Size of encoded features array.

        num_classes: int
            Number of classes to classify the encoded array into.
        """

        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(features_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.relu =  nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x