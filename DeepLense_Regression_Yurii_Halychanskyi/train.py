from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import torch
from torchvision import transforms

from models.xresnet_hybrid import xresnet_hybrid101
from utils.utils import standardize, file_path, dir_path
from utils.custom_activation_functions import Mish_layer
from utils.custom_loss_functions import root_mean_squared_error, mae_loss_wgtd
from data.custom_datasets import RegressionNumpyArrayDataset

import argparse
import numpy as np

def main(path_to_images,path_to_labels,output_dir,batch_size,mmap_mode,num_of_epochs,lr, image_dtype, image_shape, images_num):
    # Path to the dataset
    path_to_images = path_to_images
    path_to_masses = path_to_labels
    # Number of images
    # Load the dataset
    # Memmap loads images to RAM only when they are used
    images = np.memmap(path_to_images,
                       dtype=image_dtype,
                       mode='r',
                       shape=(images_num,*images_num))

    labels = np.memmap(path_to_masses,
                       dtype='float32',
                       mode='r',
                       shape=(images_num,1))

    # Split the dataset into train, valid, test subdatasets
    num_of_images = labels.shape[0]
    # 90% for train
    # 10% for valid
    max_indx_of_train_images = int(num_of_images*0.9)
    max_indx_of_valid_images = max_indx_of_train_images + int(num_of_images*0.1)
    permutated_indx = np.random.permutation(num_of_images)
    train_indx = permutated_indx[:max_indx_of_train_images]
    valid_indx = permutated_indx[max_indx_of_train_images:max_indx_of_valid_images]
    # Define transforms
    base_image_transforms = [
        transforms.Resize(image_shape)
    ]
    rotation_image_transofrms = [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 360))
    ]
    # Crete datasets
    train_dataset = RegressionNumpyArrayDataset(images, labels, train_indx, transforms.Compose(base_image_transforms+rotation_image_transofrms))
    valid_dataset = RegressionNumpyArrayDataset(images, labels, valid_indx, transforms.Compose(base_image_transforms))
    # Create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = batch_size
    dls = DataLoaders.from_dsets(train_dataset,valid_dataset,batch_size=batch_size, device=device, num_workers=2)
    # Create model
    # N_out is the number of output neurons in the last linear layer.
    # C_in is the number of channels in the input images.
    model = xresnet_hybrid101(n_out=1, sa=True, act_cls=Mish_layer, c_in=1, device=device)
    # Create learner
    learn = Learner(
        dls, 
        model,
        opt_func=ranger, 
        loss_func= root_mean_squared_error,  
        metrics=[mae_loss_wgtd],
        model_dir = output_dir
    )
    # Train the model
    num_of_epochs = num_of_epochs
    lr = lr
    learn.fit_one_cycle(num_of_epochs,lr,cbs=[
       SaveModelCallback(monitor='mae_loss_wgtd',fname='best_model', with_opt=True)])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train resnet101 to approximate the mass density of strong lensing images.')

    parser.add_argument('--image_shape', nargs='+', required=True,
                        help='Shape of a single image from the dataset.')
    parser.add_argument('--image_dtype', required=True, default='uint16',
                        help='Dtype of a signel image from the dataset.')
    parser.add_argument('--files_number', required=True,
                        help='The number of files in the dataset.',type=int)
    parser.add_argument('--path_to_images', required=True,type=file_path,
                        help='The path to a .npy file with images. It has to have the following dimensions: (num_of_elements,1,*image_shape).')
    parser.add_argument('--path_to_labels', required=True,type=file_path,
                    help='The path to a .npy file with density masses. It has to have the following dimensions: (num_of_elements,1).')
    parser.add_argument('--output_dir', required=True,type=dir_path,
                help='The directory where the best_model.pth (weights of the model) file will be stored after training.')
    parser.add_argument('--batch_size', required=True,default=64,
        help='Batch size',type=int)
    parser.add_argument('--mmap_mode',default=False,action='store_true',
        help='Use the flag if you cannot fit the whole dataset in the RAM.')
    parser.add_argument('--num_of_epochs', required=True,default=120,
        help='Number of epochs',type=int)
    parser.add_argument('--lr', required=True,default=1e-2,
        help='Learning rate',type=float)

    args = parser.parse_args()

    main(path_to_images=args.path_to_images,
        path_to_labels=args.path_to_labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        mmap_mode=args.mmap_mode,
        num_of_epochs=args.num_of_epochs,
        lr=args.lr,
        images_num=args.files_number,
        image_dtype= args.image_dtype,
        image_shape=args.image_shape)
        