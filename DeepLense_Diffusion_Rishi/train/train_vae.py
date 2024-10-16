import os
import torch

from tqdm import tqdm
from models.vae_sample import plot, sample, reconstruction



class Trainer:
    def __init__(
        self,
        model,
        config,
        dataset,
        data_processor=None,
    ):
        self.model = model
        self.n_epochs = config.opt.epochs
        self.verbose = config.verbose
        self.dataset = dataset
        #self.use_distributed = config.use_distributed
        self.device = config.device
        #self.wandb_log = config.wandb_log
        self.data_processor = data_processor
        self.plot_freq = config.data.plot_freq
        self.eval_freq = config.data.eval_freq
        
    def train(
        self,
        train_data_loader,
        test_data_loader,
        vae_loss,
        optimizer,
        scheduler,
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

        self.model.train()
        train_loss = 0.0
        total_samples = 0
        best_val_loss = float('inf')

        for epoch in range(self.n_epochs):
            print(f"Starting epoch {epoch}:")
            pbar = tqdm(train_data_loader)
            for i, (images) in enumerate(pbar):
                #print(images.shape)
                images = images.to(self.device)
                x_recon, mu, logvar = self.model(images)
                #print(x_recon.shape)
                loss = vae_loss(images, x_recon, mu, logvar)#, beta=1.0, verbose=False)
                train_loss += loss.item()*len(images)
                total_samples += len(images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            train_loss /= total_samples

        # Validation
            self.model.eval()
            with torch.no_grad():
                test_loss = 0.0
                test_samples = 0
                pbar = tqdm(test_data_loader)
                for i, (images) in enumerate(pbar):
                    
                    images = images.to(self.device)
                    x_recon, mu, logvar = self.model(images)
                    loss_test = vae_loss(images, x_recon, mu, logvar)
                    test_loss += loss_test.item()*len(images)
                    test_samples += len(images)

                test_loss /= test_samples

            print(f'Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            #print(f'Loss: {loss.item()}')

            if test_loss < best_val_loss :
                best_val_loss = test_loss
                #torch.save(self.model.state_dict(), "saved_models/ae32_log_md_bestmodel.pt")
                torch.save(self.model.state_dict(), os.path.join("saved_models",  f"new_vae_cdm_512_val.pt"))
                print(f'Epoch [{epoch+1}/{self.n_epochs}] saved best model')

            if epoch % self.plot_freq == 0: 
                sampled_images = sample(model=self.model)
                reconstruction(model=self.model, trainset=self.dataset)
                #self.diffusion.save_images(sampled_images, os.path.join("plots", f"ssl_non_lenses_{epoch}.jpg"))
                torch.save(self.model.state_dict(), os.path.join("saved_models",  f"new_vae_cdm_512_train.pt"))
                
                

            

