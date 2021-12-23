from fastai.data.core import Datasets
import torch

class RegressionNumpyArrayDataset(Datasets):
  def __init__(self,x,y,indexes=None,x_transforms_func = None):

    self.x = x[indexes]
    self.y = y[indexes]

    self.x_transforms = x_transforms_func


  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    image, label = self.x[idx], self.y[idx]
    
    image = torch.tensor(image).float()
    label = torch.tensor(label).float()

    if self.x_transforms!=None:
      image= self.x_transforms(image)


    return  image , label 
