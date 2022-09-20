from torch import nn
from torchvision.models import resnet18, resnet50

class Encoder(nn.Module):
    def __init__(self, resnet, three_channels=False, pretrained=False, features_size=256):
        """
        ResNet based neural network that receives images and encodes them into an array of size `features_size`.

        Arguments:
        ----------
        resnet: str ['18', '50']
            Kind of ResNet to be used as a backbone.

        three_channels: bool
            If True enables the network to receive three-channel images.

        pretrained: bool
            If True uses a pretrained ResNet.

        features_size: int
            Size of encoded features array.
        """

        super(Encoder, self).__init__()
        
        if resnet == '18':
            self.resnet = resnet18(pretrained=pretrained)
        elif resnet == '50':
            self.resnet = resnet50(pretrained=pretrained)

        if three_channels:
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, features_size)

    def forward(self, x):
        x = self.resnet(x)
        return x

class Classifier(nn.Module):
    def __init__(self, features_size=256, num_classes=3):
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
        self.fc = nn.Linear(features_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x