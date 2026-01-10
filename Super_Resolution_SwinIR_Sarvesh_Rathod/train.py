import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import LensingDataset
from model import build_model
import os
import numpy as np

# Configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = 'data_diff' # Created by get_data_diff.py (local folder)

def main():
    print(f"Training on {DEVICE}")
    
    # Check if data exists
    if not os.path.exists(os.path.join(DATA_DIR, 'train_HR.npy')):
        print(f"Data not found in {DATA_DIR}. Run get_data_diff.py first!")
        return

    # Load Dataset
    # Assuming get_data_diff.py creates these specific files
    full_dataset = LensingDataset(
        hr_path=os.path.join(DATA_DIR, 'train_HR.npy'),
        lr_path=os.path.join(DATA_DIR, 'train_LR.npy')
    )
    
    # Split Train/Val (90/10)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = build_model().to(DEVICE)
    
    # Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                sr = model(lr)
                loss = criterion(sr, hr)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")
        
        # Save Checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"swinir_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
