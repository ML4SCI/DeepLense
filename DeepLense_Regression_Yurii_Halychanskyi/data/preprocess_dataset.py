# -----------------------------------------------------------
# Converts multiple .npy files, where every file contains a single image and the corresponding mass,
# into a single .npy file using np.memmap.
# -----------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm
import argparse
import sys
# setting path
sys.path.append('../utils')
from utils.utils import dir_path


def main(path_to_images, result_path):
    file_names = os.listdir(path_to_images)
    files_num = len(file_names)
    img_shape = np.load(os.path.join(path_to_images, file_names[0]), allow_pickle=True)[0].shape

    # Create base mmep_map on the disk where we will be writting the files
    images_mmep = np.memmap(os.path.join(result_path,'images_mmep.npy'),
                            dtype='int16',
                            mode='w+',
                            shape=(files_num,*img_shape))

    masses_mmep = np.memmap(os.path.join(result_path,'masses_mmep.npy'),
                            dtype='float32',
                            mode='w+',
                            shape=(files_num,1))

    # Index that shows which part of the file is being written to
    w_index = 0
    for file_name in tqdm(file_names):
        img, mass = np.load(os.path.join(path_to_images, file_name),allow_pickle=True)
        images_mmep[w_index:w_index+1,:] = img
        masses_mmep[w_index:w_index+1,:] = mass
        w_index+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a multiple .npy files, where every file contains a single image and the corresponding mass, into a single .npy file using np.memmap.')

    parser.add_argument('--path_to_images', required=True,type=dir_path,
                        help='The path to a directory where multiple .npy images are located.')
    parser.add_argument('--result_path', required=True,type=dir_path,
                        help='The path where a newly created single .npy file will be stored.')

    args = parser.parse_args()

    main(path_to_images=args.path_to_images,
         result_path = args.result_path)

