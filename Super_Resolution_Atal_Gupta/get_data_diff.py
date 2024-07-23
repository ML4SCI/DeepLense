import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

mypath = './pairs'
files = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.endswith(".npy")]

HR = defaultdict()
LR = defaultdict()
HRlabels = defaultdict()
LRlabels = defaultdict()

null_list = []
for file in files:
    split = file.split("_")
    label = split[1]
    #print(label)
    if len(split) == 2:
        continue
    source = split[2].split(".")[0]
    index = int(os.path.basename(split[0]).split("_")[0])
    if index in null_list:
        continue
    if source == "hsc":
        sample = np.load(file, allow_pickle=True)
        c = 1e-7
        sample = (sample - np.amin(sample)) / (np.amax(sample) - np.amin(sample) + c)
        sample = cv2.resize(sample, (128, 128), interpolation=cv2.INTER_CUBIC)
        sample = 2*sample-1
        HR[index] = sample.reshape(1,128,128)
        HRlabels[index] = label
    elif source == "hst":
        sample = np.load(file, allow_pickle=True)
        c = 1e-7
        sample = (sample - np.amin(sample))/(np.amax(sample) - np.amin(sample)+c)
        sample = cv2.resize(sample, (64, 64), interpolation=cv2.INTER_CUBIC)
        sample = 2*sample-1
        LR[index] = sample.reshape(1,64,64)
        LRlabels[index] = label

HR = np.asarray(list(dict(sorted(HR.items())).values()))
LR = np.asarray(list(dict(sorted(LR.items())).values()))
HRlabels = np.asarray(list(dict(sorted(HRlabels.items())).values()))
LRlabels = np.asarray(list(dict(sorted(LRlabels.items())).values()))

# Find indices where NaN values are present in HR
nan_indices_in_HR = np.unique(np.where(np.isnan(HR))[0])

# Determine the indices to keep (i.e., those not containing NaN values).
indices_to_keep = np.setdiff1d(np.arange(HR.shape[0]), nan_indices_in_HR)

# Filter out the samples with NaN values in both HR and LR
HR = HR[indices_to_keep]
LR = LR[indices_to_keep]
HRlabels = HRlabels[indices_to_keep]
LRlabels = LRlabels[indices_to_keep]

print(HR.shape)
print(LR.shape)
print(HRlabels.shape)
print(LRlabels.shape)

# Plot HR sample
idx = 30
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.imshow(HR[idx].squeeze())  # .squeeze() in case the array is 4D
plt.title(f'High Resolution Sample')
plt.colorbar()

# Plot corresponding LR sample
plt.subplot(1, 2, 1)
plt.imshow(LR[idx].squeeze())  # .squeeze() in case the array is 4D
plt.title(f'Low Resolution Sample')
plt.colorbar()
plt.show()


# np.save('./data_diff/train_HR.npy', HR[:2880])
# np.save('./data_diff/test_HR.npy', HR[2880:])
# np.save('./data_diff/train_LR.npy', LR[:2880])
# np.save('./data_diff/test_LR.npy', LR[2880:])
#
# print(LRlabels[2880:])

