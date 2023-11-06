import os
import numpy as np
import cv2

def add_gaussian_blur(image, sigma):
    ksize = (int(2 * round(3*sigma) + 1), int(2 * round(3*sigma) + 1))
    return cv2.GaussianBlur(image, ksize, sigma)

def add_gaussian_noise(image, mu=0, sigma=0.1):
    noise = np.random.normal(mu, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

mypath = './Model_IV/train/no_sub'
files = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.endswith(".npy")]
Trainfiles = files[:int(len(files)*0.9)]
Testfiles = files[int(len(files)*0.9):]

train_HR = []
train_LR = []
for file in Trainfiles:
    sample = np.load(file, allow_pickle=True)
    sample = sample[0]
    sample = (sample - np.amin(sample))/(np.amax(sample) - np.amin(sample))
    sample = 2*sample-1
    sample = sample.reshape(1,64,64)
    train_HR.append(sample)
    sample_lr = add_gaussian_blur(sample.reshape(64,64), np.random.uniform(0.5,2.5))
    sample_lr = add_gaussian_noise(sample_lr, 0, np.random.uniform(0.01,0.1))
    sample_lr = 2*sample_lr-1
    sample_lr = sample_lr.reshape(1,64,64)
    train_LR.append(sample_lr)
train_HR = np.asarray(train_HR).reshape(-1,100,1,64,64)
train_LR = np.asarray(train_LR).reshape(-1,100,1,64,64)
print(train_HR.shape)
print(train_LR.shape)
os.makedirs('./Train', exist_ok=True)
np.save('./Train/train_HR.npy', train_HR)
np.save('./Train/train_LR.npy', train_LR)
print('Train Data Done!')

test_HR = []
test_LR = []
for file in Testfiles:
    sample = np.load(file, allow_pickle=True)
    sample = sample[0]
    sample = (sample - np.amin(sample))/(np.amax(sample) - np.amin(sample))
    sample = 2*sample-1
    sample = sample.reshape(1,64,64)
    test_HR.append(sample)
    sample_lr = add_gaussian_blur(sample.reshape(64,64), np.random.uniform(0.5,2.5))
    sample_lr = add_gaussian_noise(sample_lr, 0, np.random.uniform(0.01,0.1))
    sample_lr = 2*sample_lr-1
    sample_lr = sample_lr.reshape(1,64,64)
    test_LR.append(sample_lr)
test_HR = np.asarray(test_HR)
test_LR = np.asarray(test_LR)
print(test_HR.shape)
print(test_LR.shape)
os.makedirs('./Test', exist_ok=True)
np.save('./Test/test_HR.npy', test_HR)
np.save('./Test/test_LR.npy', test_LR)
print('Test Data Done!')
