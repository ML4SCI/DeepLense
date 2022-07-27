import torch
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, Flatten, Linear, ReLU, Softmax, Dropout

class BaselineModel(torch.nn.Module):
    def __init__(self, image_size=150, dropout_rate=0.5, num_classes=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer1 = Conv2d(1, 16, 5) # 1xsizexsize -> 16x(size-4)x(size-4)
        self.norm1 = BatchNorm2d(16)

        self.layer2 = Conv2d(16, 32, 3) # 16x(size-4)x(size-4) -> 32x(size-8)x(size-8)
        self.norm2 = BatchNorm2d(32)
        self.pool2 = MaxPool2d(2) # 32x(size-8)x(size-8) -> 32x(size/2-4)x(size/2-4)

        self.layer3 = Conv2d(32, 32, 3) # 32x(size/2-4)x(size/2-4) -> 32x(size/2-6)x(size/2-6)
        self.norm3 = BatchNorm2d(32)
        self.pool3 = MaxPool2d(2) # 32x(size/2-6)x(size/2-6) -> 32x(size/4 - 3)x(size/4 - 3)

        self.layer4 = Conv2d(32, 32, 3) # 32x(size/4 - 3)x(size/4 - 3) -> 32x(size/4 - 5)x(size/4 - 5)
        self.norm4 = BatchNorm2d(32)
        self.pool4 = MaxPool2d(2) # 32x(size/4 - 5)x(size/4 - 5) -> 32x(size/8 - 5/2)x(size/8 - 5/2)

        self.resid5 = Conv2d(32, 64, 1)
        self.layer5 = Conv2d(32, 64, 3, padding='same') # 32x(size/8 - 5/2)x(size/8 - 5/2) -> 64x(size/8 - 5/2)x(size/8 - 5/2)
        self.norm5 = BatchNorm2d(64)

        self.resid6 = Conv2d(64, 64, 1)
        self.layer6 = Conv2d(64, 64, 3, padding='same') # 64x(size/8 - 5/2)x(size/8 - 5/2) -> 64x(size/8 - 5/2)x(size/8 - 5/2)
        self.norm6 = BatchNorm2d(64)

        self.layer7 = Conv2d(64, 128, 3) # 64x(size/8 - 5/2)x(size/8 - 5/2) -> 128x(size/8 - 5/2 - 2)x(size/8 - 5/2 - 2)
        self.norm7 = BatchNorm2d(128)

        self.flatten = Flatten()

        repr_size = int((image_size / 8) - 4.5)

        self.linear1 = Linear(128*repr_size*repr_size, 256)
        self.linear2 = Linear(256, 32)

        self.classifier = Linear(32, num_classes)

        self.pool = MaxPool2d(2)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        self.dropout = Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.pool2(self.relu(self.norm2(self.layer2(x))))
        x = self.pool3(self.relu(self.norm3(self.layer3(x))))
        x = self.pool4(self.relu(self.norm4(self.layer4(x))))

        x = self.relu(self.norm5(self.layer5(x))) + self.resid5(x) # Skip connection
        x = self.relu(self.norm6(self.layer6(x))) + self.resid6(x) # Skip connection

        x = self.relu(self.norm7(self.layer7(x)))

        x = self.dropout(self.flatten(x))

        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

        logits = self.classifier(x)

        return logits
    
    def predict(self, x):
        logits = self(x)
        probs = self.softmax(logits)
        return torch.argmax(probs, dim=-1)