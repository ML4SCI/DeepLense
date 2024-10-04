import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from model import MySCNN

# Custom Dataset for loading Laplacians, Boundaries, and Labels
class SCNNDataset(Dataset):
    def __init__(self, laplacians_file, boundaries_file, labels_file):
        self.laplacians = np.load(laplacians_file, allow_pickle=True)
        self.boundaries = np.load(boundaries_file, allow_pickle=True)
        self.labels = np.load(labels_file, allow_pickle=True)

    def __len__(self):
        return len(self.labels.files)

    def __getitem__(self, idx):
        lap = self.laplacians[f'arr_{idx}']
        bounds = self.boundaries[f'arr_{idx}']
        label = 1 if self.labels[f'arr_{idx}'][0] == 1 else 0
        return lap, bounds, torch.tensor(label, dtype=torch.float).unsqueeze(0)


# Define the Lightning Module
class MyLightningModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(MyLightningModel, self).__init__()
        self.network = MySCNN(colors=1)  # Your SCNN model
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate

        # For storing losses and accuracy
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        self.losses = {'train': [], 'val': [], 'test': []}

    def forward(self, Ls, Ds, adDs, xs):
        return self.network(Ls, Ds, adDs, xs)

    def training_step(self, batch, batch_idx):
        laplacians, boundaries, labels = batch

        # Convert laplacians and boundaries to tensors (adapt to your `scnn.coo2tensor` as needed)
        Ls = [scnn.coo2tensor(chebyshev.normalize(lap[k])) for k in range(len(laplacians))]
        Ds = [scnn.coo2tensor(boundaries[k].transpose()) for k in range(len(boundaries))]
        adDs = [scnn.coo2tensor(boundaries[k]) for k in range(len(boundaries))]

        # Example features input (adapt to your input pipeline)
        xs = [torch.rand(1, 4, laplacians[0].shape[0])]  # Modify to fit your model input shape

        # Forward pass
        probs, _ = self.forward(Ls, Ds, adDs, xs)
        loss = self.criterion(probs, labels.unsqueeze(0))

        # Log and store losses
        self.losses['train'].append(loss.item())
        self.log('train_loss', loss)

        # Compute accuracy
        preds = torch.round(probs)
        acc = (preds == labels).float().mean()
        self.train_acc.append(acc.item())
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        laplacians, boundaries, labels = batch

        Ls = [scnn.coo2tensor(chebyshev.normalize(lap[k])) for k in range(len(laplacians))]
        Ds = [scnn.coo2tensor(boundaries[k].transpose()) for k in range(len(boundaries))]
        adDs = [scnn.coo2tensor(boundaries[k]) for k in range(len(boundaries))]
        xs = [torch.rand(1, 4, laplacians[0].shape[0])]

        # Forward pass
        probs, _ = self.forward(Ls, Ds, adDs, xs)
        loss = self.criterion(probs, labels.unsqueeze(0))

        # Log and store validation losses
        self.losses['val'].append(loss.item())
        self.log('val_loss', loss)

        # Compute accuracy
        preds = torch.round(probs)
        acc = (preds == labels).float().mean()
        self.val_acc.append(acc.item())
        self.log('val_acc', acc)

        return loss

    def test_step(self, batch, batch_idx):
        laplacians, boundaries, labels = batch

        Ls = [scnn.coo2tensor(chebyshev.normalize(lap[k])) for k in range(len(laplacians))]
        Ds = [scnn.coo2tensor(boundaries[k].transpose()) for k in range(len(boundaries))]
        adDs = [scnn.coo2tensor(boundaries[k]) for k in range(len(boundaries))]
        xs = [torch.rand(1, 4, laplacians[0].shape[0])]

        # Forward pass
        probs, _ = self.forward(Ls, Ds, adDs, xs)
        loss = self.criterion(probs, labels.unsqueeze(0))

        # Log and store test losses
        self.losses['test'].append(loss.item())
        self.log('test_loss', loss)

        # Compute accuracy
        preds = torch.round(probs)
        acc = (preds == labels).float().mean()
        self.test_acc.append(acc.item())
        self.log('test_acc', acc)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_end(self):
        # Save the losses and accuracy metrics after training
        np.savez_compressed("training_stats.npz", train_acc=self.train_acc, val_acc=self.val_acc,
                            test_acc=self.test_acc, train_loss=self.losses['train'],
                            val_loss=self.losses['val'], test_loss=self.losses['test'])


# Create DataLoaders
def get_dataloader(lapl_file, bounds_file, labels_file, batch_size=32, shuffle=True):
    dataset = SCNNDataset(lapl_file, bounds_file, labels_file)
    for x in dataset:
        print(x)
        input()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    os.chdir(os.path.abspath('bounds_and_laps/'))
    # Training, Validation, and Test Dataloaders

    train_loader = get_dataloader('train_laplacians.npz', 'train_boundaries.npz', 'train_labels.npz')
    val_loader = get_dataloader('val_laplacians.npz', 'val_boundaries.npz', 'val_labels.npz', shuffle=False)
    test_loader = get_dataloader('test_laplacians.npz', 'test_boundaries.npz', 'test_labels.npz', shuffle=False)


    # Lightning Trainer
    trainer = pl.Trainer(max_epochs=10)
    model = MyLightningModel()

    trainer.fit(model, train_loader, val_loader)  # Train and validate
    trainer.test(model, test_loader)  # Test
