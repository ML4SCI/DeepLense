from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
import torch
from torchvision import transforms

from models.xresnet_hybrid import xresnet_hybrid101
from utils.utils import standardize, file_path,dir_path
from utils.custom_activation_functions import Mish_layer


import argparse
import numpy as np
from tqdm import tqdm


def main(path_to_images,path_to_weights,output_dir,mmap_mode):
    # Path to the dataset
    path_to_images = path_to_images
    # Load the dataset
    images = np.load(path_to_images,mmap_mode='r' if mmap_mode else None).astype('float32')
    images = images.reshape(-1,1,150,150)
    # Calculate the stats of the dataset to standardize it
    IMAGES_MEAN, IMAGES_STD = images.mean(), images.std()
    images = standardize(images,IMAGES_STD,IMAGES_MEAN)
    # Define transforms
    resize_transf =  transforms.Resize(150)
    # Create dataloader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create model
    torch.manual_seed(50)
    # N_out is the number of output neurons in the last linear layer.
    # C_in is the number of channels in the input images.
    model = xresnet_hybrid101(n_out=1, sa=True, act_cls=Mish_layer, c_in=1,device=device)
    # Load the weights
    model.load_state_dict(torch.load(path_to_weights))
    # Transfer parameters to GPU if available
    model = model.to(device)
    # Enter Evaluation mode
    model.eval()
    # Stop calculating gradients
    with torch.no_grad():
        predictions = []
        for image in tqdm(images):
            image_tensor = resize_transf(torch.tensor(image,dtype=torch.float)).reshape(1,1,150,150).to(device)
            predictions.append(model(image_tensor).cpu().detach().numpy())

    predictions = np.concatenate(predictions,axis=0)
    full_path = os.path.join(output_dir,'predicted_mass_densities.npy')
    # Save the output
    np.save(full_path,predictions)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Predict density masses for the dataset by using trained resnet101.')

    parser.add_argument('--path_to_images', required=True,type=file_path,
                        help='The path to a .npy file with images. It has to have the following dimensions: (num_of_elements,1,150,150).')
    parser.add_argument('--path_to_weights', required=True,type=file_path,
                    help='The path to a .pth file with trained weights for XResnetHybrid101.')
    parser.add_argument('--mmap_mode',default=False,action='store_true',
        help='Use the flag if you cannot fit the whole dataset in the RAM.')
    parser.add_argument('--output_dir', required=True,type=dir_path,
                help='The directory where the model will output predicted mass densities in a .npy file.')

    args = parser.parse_args()
    
    main(path_to_images=args.path_to_images,
        path_to_weights=args.path_to_weights,
        output_dir=args.output_dir,
        mmap_mode=args.mmap_mode)