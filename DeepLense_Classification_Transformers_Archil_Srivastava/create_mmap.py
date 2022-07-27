import numpy as np
import argparse
import os
from tqdm import tqdm

from constants import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/Model_I', help='root directory for dataset')
    parser.add_argument('--out-dir', default='/data/Model_I/memmap', help='directory to store memmap')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--image_size', type=int, default=150, help='Input image size')
    opt = parser.parse_args()
    if opt.mode not in ['train', 'test']:
        raise(ValueError('Mode should be either train or test'))

    os.system(f'mkdir {opt.out_dir}')
    os.system(f'mkdir {os.path.join(opt.out_dir, opt.mode)}')

    train_size = 0
    for class_dir in os.scandir(os.path.join(opt.data_dir, opt.mode)):
        raw_image_files = os.listdir(class_dir.path)
        train_size += len(raw_image_files)
    
    image_map = np.memmap(os.path.join(opt.out_dir, opt.mode, 'images.npy'),
                          dtype='int32', mode='w+',
                          shape=(train_size, opt.image_size, opt.image_size))
    labels = np.zeros(train_size, dtype='int32')

    idx = 0
    for class_dir in tqdm(os.scandir(os.path.join(opt.data_dir, opt.mode))):
        label = LABEL_MAP[class_dir.name]
        for raw_image_file in tqdm(os.scandir(class_dir.path), desc=class_dir.name):
            if class_dir.name == 'axion': # There is mass float value in axion data
                img, _ = np.load(raw_image_file.path, allow_pickle=True)
            else:
                img = np.load(raw_image_file)
            
            image_map[idx:idx+1] = img
            labels[idx] = label
            idx += 1
        image_map.flush()
    np.save(os.path.join(opt.out_dir, opt.mode, 'labels.npy'), labels)
