import os
import numpy as np
import cv2
from tqdm import tqdm

# Folder containing HR images from DeepLenseSim
HR_FOLDER = "hr_images"
OUTPUT_FOLDER = "pairs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def degrade_image(hr_img):
    # Apply Gaussian blur
    lr_img = cv2.GaussianBlur(hr_img, (5, 5), 0)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.02, hr_img.shape)
    lr_img = lr_img + noise

    # Clip values
    lr_img = np.clip(lr_img, 0, 1)
    return lr_img

files = sorted(os.listdir(HR_FOLDER))

for i, file in enumerate(tqdm(files)):
    if file.endswith(".npy"):
        hr = np.load(os.path.join(HR_FOLDER, file))

        lr = degrade_image(hr)

        np.save(f"{OUTPUT_FOLDER}/{i}_label_hsc.npy", hr)
        np.save(f"{OUTPUT_FOLDER}/{i}_label_hst.npy", lr)

print("Dataset generation complete!")
