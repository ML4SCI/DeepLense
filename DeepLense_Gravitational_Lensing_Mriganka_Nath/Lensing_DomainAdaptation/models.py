import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # reqd for pretraining images
from e2cnn import gspaces
import e2cnn.nn as e2nn

# pretrained models that can be used as backbone for differnt DA techniques
pretrained_model1 = 'tf_efficientnet_b2_ns'
pretrained_model2 = 'resnet34d'
pretrained_model3 = 'densenet121'

p = []
p.append(pretrained_model1) 
p.append(pretrained_model2) 
p.append(pretrained_model3) 

def available_backbone_models():
    for _ , pp in enumerate(p):
        print(f'Model Name : {pp}') 

# defines the size of the vector outputted by the encoder
latent_size = 256

class Encoder(nn.Module):
    def __init__(self, model_name = pretrained_model1,latent_size =  latent_size , pretrained = True , dropout_rate = 0.5):
        super().__init__()
        self.m_name = model_name
        if( self.m_name == pretrained_model1):
            num_channels = 1408 #for effnet
        elif (self.m_name == pretrained_model2):
            num_channels = 512 #for resnet
        else :
            num_channels = 1024 #for densenet
        self.backbone = timm.create_model( self.m_name, pretrained=pretrained, num_classes=0,global_pool='',in_chans=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.prelu = nn.PReLU() 
        self.lin = nn.Linear( num_channels, latent_size)
        self.do = nn.Dropout(p= dropout_rate)
        
    def forward(self,image):
        image = self.backbone(image)     
        image = self.pool(image)
        image = image.view(image.shape[0], -1)    
        image = self.do(image)
        image = self.prelu(self.lin(image))
        return image

class ECNN(nn.Module):
    def __init__(self, features_size= latent_size, sym_group='Dihedral', N=2):
        """
        taken from https://arxiv.org/abs/2112.12121
        """

        super(ECNN, self).__init__()

        if sym_group == 'Dihedral':
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)

        elif sym_group == 'Circular':
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        in_type = e2nn.FieldType(self.r2_act, 1*[self.r2_act.trivial_repr])
        self.input_type = in_type

        out_type = e2nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.MaskModule(in_type, 64, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block1.out_type
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block2.out_type
        out_type = e2nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block3.out_type
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        in_type = self.block4.out_type
        out_type = e2nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        in_type = self.block5.out_type
        out_type = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.gpool = e2nn.GroupPooling(out_type)
        c = self.gpool.out_type.size

        self.fully_net = nn.Sequential(
            nn.Linear(5184, features_size),
            nn.BatchNorm1d(features_size),
            nn.ELU(inplace=True),
            nn.Linear(features_size, features_size),
        )

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.gpool(x)
        x = x.tensor
        x = self.fully_net(x.reshape(x.shape[0], -1))
        return x

class Classifier(nn.Module):
    def __init__(self,latent_size =  latent_size):
        super().__init__()
        self.lin = nn.Linear(  latent_size,1)
    
    def forward(self,image):        
        image = self.lin(image)
        return image

# required for ADDA
class Discriminator(nn.Module):
    def __init__(self,latent_size = latent_size):
        super().__init__()
        self.discrim = nn.Sequential(
            nn.Linear(latent_size, latent_size // 2),
            nn.PReLU(),
            nn.Linear(latent_size//2, latent_size//2),
            nn.PReLU(),
            nn.Linear(latent_size//2, 1)
        )
    
    def forward(self, x):
        x = self.discrim(x)
        return x
