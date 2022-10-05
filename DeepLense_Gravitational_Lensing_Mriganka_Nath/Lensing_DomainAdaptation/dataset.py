import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

def make_labels(paths,target):
    xx = np.array(paths)
    xx = np.expand_dims(xx,1)
    yy = np.zeros((len(xx),1)) + target
    gg = np.concatenate([xx,yy],axis = 1)
    return gg

def prep_data(class1 , class2):
    # class1 is non_lensing images
    # class2 is lensed images
    d1 = make_labels(class1 , 0)
    d2 = make_labels(class2 , 1)
    t_data = np.concatenate([d1,d2] , axis = 0)
    X, X_test = train_test_split(t_data, test_size=0.1, random_state=42 , stratify = t_data[:,1] ) # 10% set to test data
    X_train, X_val = train_test_split(X, test_size=0.25, random_state=42 ,stratify = X[:,1])

    return X_train, X_val , X_test

class Len(Dataset):
    '''
    Dataset class

    Arguments:
    _________

    data: The image data containing the 
        path and the label
    
    augs: The augmentation you will 
        provide

    Returns:
    _______

    image: a three dimensional image 
        as a torch tensor 

    target: The target (0/1) as a 
        torch tensor depending on lensed/non-lensed

    '''
    def __init__(self , data , augs):
        self.data = data
        self.augs = augs
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self , idx):
        path = self.data[idx][0] 
        target = float(self.data[idx][1])

        image = np.load(path)      
        image = (image - np.min(image))/(np.max(image) - np.min(image)) #make the values raof image range from 0 to1
        image = np.expand_dims(image , axis = 2)
        
        transformed = self.augs(image=image)       
        image = transformed['image']
        image = torch.tensor(image,dtype = torch.float32)      
         
        return image,torch.tensor(target).long()

class Discriminator_dataset(Dataset):
    '''
        Dataset class usind in ADDA

        Arguments:
        _________

        data: The image data containing the 
            path and the label

        augs: The augmentation you will 
            provide
        
        source: if the image belong to source 
            or target dataset

        Returns:
        _______

        image: a three dimensional image 
            as a torch tensor 

        target: The target (0/1) as a torch 
            tensor depending on belonging from source/target dataset

    '''
    def __init__(self , data , augs , source = True):
        self.data = data
        self.augs = augs
        self.source = source
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self , idx):
        path = self.data[idx][0] 

        image = np.load(path)      
        image = (image - np.min(image))/(np.max(image) - np.min(image)) #make the values raof image range from 0 to1
        image = np.expand_dims(image , axis = 2)
        
        transformed = self.augs(image=image)       
        image = transformed['image']
        image = torch.tensor(image,dtype = torch.float32)
        
        target = 0.
        if (self.source == False):
            target = 1.
            
         
        return image,torch.tensor(target).long()

class SE_data(Dataset):
    '''
    Dataset class used in self-ensembling

    Arguments:
    _________

    data: The image data containing the 
        path and the label
    
    augs: The augmentation you will 
        provide

    Returns:
    _______

    (same image is outputted with differnt augmentations)
    image1: a three dimensional image 
        as a torch tensor 

    image2: a three dimensional image 
        as a torch tensor
            
    '''
    def __init__(self , data , augs ):
        self.data = data
        self.augs = augs
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self , idx):
        path = self.data[idx][0] 

        image = np.load(path)      
        image = (image - np.min(image))/(np.max(image) - np.min(image)) #make the values raof image range from 0 to1
        image = np.expand_dims(image , axis = 2)
        image1 = image
        image2 = image
        
        transformed1 = self.augs(image=image1)       
        image1 = transformed1['image']
        image1 = torch.tensor(image1,dtype = torch.float32)
        
        transformed2 = self.augs(image=image2)       
        image2 = transformed2['image']
        image2 = torch.tensor(image2,dtype = torch.float32)

        return image1,image2

class AMatch_data(Dataset):
    '''
    Dataset class used in AdaMatch

    Arguments:
    _________

    data: The image data containing the 
        path and the label
    
    augs1: weak augmentations

    augs2: strong augmentations 

    Returns:
    _______

    (same image is outputted with weak/strong augmentations)
    image1: a three dimensional image 
        as a torch tensor 

    image2: a three dimensional image 
        as a torch tensor

    target: The target (0/1) as a 
        torch tensor depending on lensed/non-lensed
            
    '''
    def __init__(self , data , augs1 , augs2 ):
        self.data = data
        self.augs1 = augs1 #weak_aug
        self.augs2 = augs2 #strong_aug
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self , idx):
        path = self.data[idx][0]
        target = float(self.data[idx][1])

        image = np.load(path)      
        image = (image - np.min(image))/(np.max(image) - np.min(image)) #make the values raof image range from 0 to1
        image = np.expand_dims(image , axis = 2)
        image1 = image
        image2 = image
        
        transformed1 = self.augs1(image=image1)       
        image1 = transformed1['image']
        weak_image = torch.tensor(image1,dtype = torch.float32)
        
        transformed2 = self.augs2(image=image2)       
        image2 = transformed2['image']
        strong_image = torch.tensor(image2,dtype = torch.float32)
        
        return weak_image,strong_image,target
