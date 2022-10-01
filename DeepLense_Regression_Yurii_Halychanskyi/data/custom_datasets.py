from fastai.data.core import Datasets
import torch
import numpy as np

class RegressionNumpyArrayDataset(Datasets):
  def __init__(self,x,y,indexes=None,x_transforms_func = None):

    self.x = x[indexes]
    self.y = y[indexes]

    self.x_transforms = x_transforms_func


  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    image, label = self.x[idx], self.y[idx]

    img_min = image.min()
    image = torch.tensor((image-img_min)/(image.max()-img_min), dtype=torch.float32)
    label = torch.tensor(np.log10(label), dtype=torch.float32)

    if self.x_transforms!=None:
      image= self.x_transforms(image)


    return  image , label 
