import os
import torch

from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        config,
        data_processor=None,
    ):
        self.model = model
        self.n_epochs = config.opt.epochs
        #self.use_distributed = config.use_distributed
        self.device = config.device
        #self.wandb_log = config.wandb_log
        self.data_processor = data_processor

    def train(
        self,
        train_data_loader,
        test_data_loader,
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
        self.model.train()
        train_loss = 0.0
        total_samples = 0
        best_val_loss = float('inf')

        for epoch in range(self.n_epochs):

            print(f"Starting epoch {epoch}:")
            pbar = tqdm(train_data_loader)
            for i, (images) in enumerate(pbar):
                
                images = images.to(self.device)
                images = images.to(torch.float)
                output = self.model(images)
                loss = mse(images, output)
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
                    output = self.model(images)
                    loss_test = mse(images, output)
                    test_loss += loss_test.item()*len(images)
                    test_samples += len(images)

                test_loss /= test_samples

            print(f'Epoch [{epoch+1}/{self.n_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

            if test_loss < best_val_loss :
                best_val_loss = test_loss
                torch.save(self.model.state_dict(), "saved_models/ae32_log_md_bestmodel.pt")
                print(f'Epoch [{epoch+1}/{self.n_epochs}] saved best model')
            


