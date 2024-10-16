import torch
import numpy as np
import os

import torchvision.transforms as Transforms
import matplotlib.pyplot as plt

#root_dir = '../Data/cdm_regress_multi_param_model_ii/cdm_regress_multi_param/'
#root_dir = '../Data/npy_lenses-20240731T044737Z-001/npy_lenses/'
root_dir = '../Data/real_lenses_dataset/lenses'
data_list_cdm = [ f for f in os.listdir(root_dir) if f.endswith('.npy')]
#print(data_list_cdm)
data_file_path = os.path.join(root_dir, data_list_cdm[50])
data = np.load(data_file_path)#, allow_pickle=True)
print(data.shape)
data = (data - np.min(data))/(np.max(data)-np.min(data))
print(np.min(data))
print(np.max(data))

transforms = Transforms.Compose([
                # Transforms.ToTensor(), # npy loader returns torch.Tensor
                Transforms.CenterCrop(64),
                #Transforms.Normalize(mean = [0.06814773380756378, 0.21582692861557007, 0.4182431399822235],\
                 #                       std = [0.16798585653305054, 0.5532506108283997, 1.1966736316680908]),
            ]) 

data_torch = torch.from_numpy(data)
data_torch = transforms(data_torch)
# print(torch.min(data_torch))
# print(torch.max(data_torch))
data_torch = data_torch.permute(1, 2, 0).to('cpu').numpy()
plt.imshow(data_torch)
plt.savefig(os.path.join("plots", f"ddpm_ssl_actual.jpg"))