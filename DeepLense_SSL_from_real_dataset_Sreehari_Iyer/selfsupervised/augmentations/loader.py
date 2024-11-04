import random
import math
import numpy as np
import torch
import os
import pickle

from torch.utils.data import Dataset, DataLoader
from typing import Union, Tuple, List, Optional, Callable, Dict

def npy_loader(path):
    sample = torch.from_numpy(np.load(path).astype(np.float32))
    return sample

class ImageDataset(Dataset):
    def __init__(
            self, 
            image_paths: List[str],
            labels: List[int],
            loader: Callable=npy_loader, 
            transform=None,
            return_indices=False):
        self.image_paths = image_paths
        self.label = labels
        self.transform = transform
        self.loader = loader
        self.return_idx=return_indices

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.loader(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.return_idx:
            return image, self.label[idx], idx
        else:
            return image, self.label[idx]

def get_dataloaders(
        data_path: str,
        train_test_indices: List,
        state: Dict,
        eval_transforms,
    ):
    indices = None
    with open(train_test_indices, "rb") as f:
        indices = pickle.load(f)

    all_paths = [os.path.join(data_path, *["lenses", img]) for img in os.listdir(os.path.join(data_path, "lenses"))] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in os.listdir(os.path.join(data_path, "nonlenses"))]
    all_labels = [0]*len(os.listdir(os.path.join(data_path, "lenses"))) + [1]*len(os.listdir(os.path.join(data_path, "nonlenses")))
    lens_train = state["history"]["lens_train_indices"]
    nonlens_train = state["history"]["nonlens_train_indices"]
    train_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(indices["train"]["lenses"])[lens_train]] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(indices["train"]["nonlenses"])[nonlens_train]]
    train_labels = [0]*len(lens_train) + [1]*len(nonlens_train)
    
    lens_val = state["history"]["lens_val_indices"]
    nonlens_val = state["history"]["nonlens_val_indices"]
    val_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(indices["train"]["lenses"])[lens_val]] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(indices["train"]["nonlenses"])[nonlens_val]]
    val_labels = [0]*len(lens_val) + [1]*len(nonlens_val)

    test_paths = [os.path.join(data_path, *["lenses", img]) for img in np.array(indices["test"]["lenses"])] +\
                        [os.path.join(data_path, *["nonlenses", img]) for img in np.array(indices["test"]["nonlenses"])]
    test_labels = [0]*len(indices["test"]["lenses"]) + [1]*len(indices["test"]["nonlenses"])

    dataset = ImageDataset(
            image_paths=all_paths,
            labels=all_labels,
            loader=npy_loader,
            transform=eval_transforms
    )
    train_dataset = ImageDataset(
            image_paths=train_paths,
            labels=train_labels,
            loader=npy_loader,
            transform=eval_transforms
    )
    val_dataset = ImageDataset(
            image_paths=val_paths,
            labels=val_labels,
            loader=npy_loader,
            transform=eval_transforms
    )
    test_dataset = ImageDataset(
            image_paths=test_paths,
            labels=test_labels,
            loader=npy_loader,
            transform=eval_transforms
    )
    data_loader = DataLoader(
                        dataset,
                        batch_size=64,
                        num_workers=0,
                        shuffle=False,
                    )
    train_loader = DataLoader(
                        train_dataset,
                        batch_size=64,
                        num_workers=0,
                        shuffle=True,
                    )
    val_loader = DataLoader(
                        val_dataset,
                        batch_size=64,
                        num_workers=0,
                        shuffle=False,
                    )
    test_loader = DataLoader(
                        test_dataset,
                        batch_size=64,
                        num_workers=0,
                        shuffle=False,
                    )
    return data_loader, train_loader, val_loader, test_loader

# ImageDatasetMasked is based on the following implementation
# https://github.com/bytedance/ibot/blob/main/loader.py
class ImageDatasetMasked(Dataset):
    def __init__(
            self, 
            image_paths: List[str],
            labels: List[int],
            loader: Callable, 
            transform, 
            patch_size, 
            pred_ratio, 
            pred_ratio_var, 
            pred_aspect_ratio, 
            pred_shape='block', 
            pred_start_epoch=0, 
            **kwargs
        ):
        super(ImageDatasetMasked, self).__init__(**kwargs)
        self.image_paths = image_paths
        self.label = labels
        self.transform = transform
        assert self.transform is not None
        self.loader = loader
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        # print(pred_aspect_ratio)
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def __len__(self):
        return len(self.image_paths)
        
    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = self.loader(img_path)
        # print(image.shape)
        # if self.transform:
        image = self.transform(image)
        # print(image.shape)
        output = (image, self.label[index])
                
        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (image, self.label[index], masks)
