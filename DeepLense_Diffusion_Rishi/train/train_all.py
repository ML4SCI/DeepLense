import os
import torch

from tqdm import tqdm
from models.ddpm_all import Diffusion

class Trainer:
    def __init__(
        self,
        model,
        config,
        data_processor=None,
    ):
        self.model = model
        self.n_epochs = config.opt.epochs
        self.verbose = config.verbose
        #self.use_distributed = config.use_distributed
        self.device = config.device
        #self.wandb_log = config.wandb_log
        self.data_processor = data_processor
        self.plot_freq = config.data.plot_freq
        self.eval_freq = config.data.eval_freq
        self.diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)

    def train(
        self,
        data_loader,
        mse,
        optimizer,
        scheduler,
        training_loss=None,
        eval_loss=None,
    ):
        """Trains the given model on the given datasets.
        params:
        data_loader: torch.utils.data.DataLoader
             dataloader
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        """

        for epoch in range(self.n_epochs):
            print(f"Starting epoch {epoch}:")
            pbar = tqdm(data_loader)
            for i, (images, v1, v2, v3, v4) in enumerate(pbar):
                #print(images)
                images = images.to(self.device)
                v1 = v1.to(self.device)
                v2 = v2.to(self.device)
                v3 = v3.to(self.device)
                v4 = v4.to(self.device)
                t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)
                predicted_noise = self.model(x_t, t, v1, v2, v3, v4)
                loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if epoch % self.plot_freq == 0: 
                sampled_images = self.diffusion.sample_conditional(self.model, n=images.shape[0], v1=v1, v2=v2, v3=v3, v4=v4)
                self.diffusion.save_images(sampled_images, os.path.join("plots", f"all_conditional_{epoch}.jpg"))
                torch.save(self.model.state_dict(), os.path.join("saved_models",  f"all_conditional_ckpt_model2.pt"))

            # if epoch % self.eval_freq == 0:
            #     FID_Score = self.diffusion.cal_fid(self.model, data_loader, self.device)
            #     print("FID score: ", FID_Score)

