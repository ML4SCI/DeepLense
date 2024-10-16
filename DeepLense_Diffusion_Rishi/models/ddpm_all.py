import os
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

from tqdm import tqdm 
from torchmetrics.image.fid import FrechetInceptionDistance
from dataset.preprocessing_model_2 import CustomDataset, CustomDataset_Conditional

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        return x

    def sample_conditional(self, model, n, v1, v2, v3, v4, cfg_scale=0):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, v1, v2, v3, v4)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        return x

    def cal_fid(self, model, train_dl, device):
        fid = FrechetInceptionDistance(feature=2048, reset_real_features = False, normalize = True).to(device)
        model.eval()
        n = 100
        with torch.no_grad():
            fake_imgs = self.sample(model, n)

        image_list = []
        num_images_to_sample = 100  # The number of images you want to sample
        total_sampled = 0
        
        # Sample a specific number of images (1000 in this example)
        for i, data in enumerate(train_dl):
            images = data
            image_list.append(images)
            total_sampled += images.size(0)  
            if total_sampled >= num_images_to_sample:
                break
        
        # Concatenate the list of images into a single tensor
        image_tensor = torch.cat(image_list[:num_images_to_sample], dim=0)
        real_imgs = image_tensor[0:100, :, :, :]
        
        real_imgs = real_imgs.to(device)
        real_imgs_rgb = self.convert_to_rgb(real_imgs, device)  # Convert to RGB
        fake_imgs_rgb = self.convert_to_rgb(fake_imgs, device)  # Convert to RGB
        fid.update(real_imgs_rgb, real=True)
        fid.update(fake_imgs_rgb, real=False)
        score = fid.compute()
        return score
    
    def cal_fid_conditional(self, model, labels, train_dl, device):
        fid = FrechetInceptionDistance(feature=2048, reset_real_features = False, normalize = True).to(device)
        model.eval()
        n = 100
        with torch.no_grad():
            fake_imgs = self.sample_conditional(model, n, labels)

        image_list = []
        num_images_to_sample = 100  # The number of images you want to sample
        total_sampled = 0
        
        # Sample a specific number of images (1000 in this example)
        pbar = tqdm(train_dl)
        for i, data in enumerate(pbar):
            #print(data)
            images = data
            image_list.append(images)
            total_sampled += images.size(0)  
            if total_sampled >= num_images_to_sample:
                break
        
        # Concatenate the list of images into a single tensor
        image_tensor = torch.cat(image_list[:num_images_to_sample], dim=0)
        real_imgs = image_tensor[0:100, :, :, :]
        
        real_imgs = real_imgs.to(device)
        real_imgs_rgb = self.convert_to_rgb(real_imgs, device)  # Convert to RGB
        fake_imgs_rgb = self.convert_to_rgb(fake_imgs, device)  # Convert to RGB
        fid.update(real_imgs_rgb, real=True)
        fid.update(fake_imgs_rgb, real=False)
        score = fid.compute()
        return score


    def convert_to_rgb(self, images, device):
    
        colormap = cm.viridis
        input_rgb_list = []
        for image in images:
            # Apply colormap
            image = image.cpu()
            input_rgb = colormap(image.numpy())  
            # Keep only RGB channels
            input_rgb = input_rgb[0, :, :, :3]
            # Convert numpy array back to tensor and permute dimensions to (channels, height, width)
            input_rgb_tensor = torch.from_numpy(input_rgb.astype(np.float32)).permute(2, 0, 1)
            input_rgb_list.append(input_rgb_tensor)
        
        # Stack the list of tensors along the batch dimension
        input_rgb_batch = torch.stack(input_rgb_list, dim=0).to(device)
    
        return input_rgb_batch
        
    def save_images(self, images, path):
        grid = torchvision.utils.make_grid(images)
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        plt.imshow(ndarr)
        plt.savefig(path)

    def cal_fid_all(self, model, data_dir, device):
        no_sub_folder = 'no_sub'
        axion_folder = 'axion'
        cdm_folder = 'cdm'
        no_sub = os.path.join(data_dir, no_sub_folder)
        axion  = os.path.join(data_dir, axion_folder)
        cdm    = os.path.join(data_dir, cdm_folder)

        # Axion is 0 .  No sub 0
        # CDM is 1, cdm 2
        # No sub is 2, Axion 1
        custom_data_set_no_sub = CustomDataset(root_dir=no_sub)
        #custom_data_set_axion  = CustomDataset(root_dir=axion)
        #custom_data_set_cdm    = CustomDataset(root_dir=cdm)

        dataloader_nosub  = DataLoader(custom_data_set_no_sub, batch_size=32, shuffle=False)
        #dataloader_axion  = DataLoader(custom_data_set_axion, batch_size=32, shuffle=False)
        #dataloader_cdm  = DataLoader(custom_data_set_cdm, batch_size=32, shuffle=False)
        
        labels_nosub = torch.full([100,], 2, dtype=torch.long).to(device)
        #labels_axion = torch.zeros([100,], dtype=torch.long).to(device)
        #labels_cdm = torch.ones([100,], dtype=torch.long).to(device)

        score_nosub = self.cal_fid_conditional(model, labels_nosub, dataloader_nosub, device)
        #score_axion = self.cal_fid_conditional(model, labels_axion, dataloader_axion, device)
        #score_cdm   = self.cal_fid_conditional(model, labels_cdm, dataloader_cdm, device)

        print("FID score for nosub :", score_nosub)
        #print("FID score for axion :", score_axion)
        #print("FID score for cdm :", score_cdm)
        


# def Plot(x_sampled,fig_size = (6,6)):
#     grid = make_grid(x_sampled.cpu(), nrow=10)
#     if fig_size is not None:
#         plt.figure(figsize= fig_size)
#     plt.imshow(grid.permute(1,2,0).detach().numpy(), cmap='gray')
#     plt.show()