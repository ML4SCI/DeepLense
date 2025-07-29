import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
from dataset import NpyDataset, classifier_transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision('high')

DATA_DIR = "/speech/advait/rooshil/nanoDiT/Model_II_normalized"
NUM_CLASSES = 3
BATCH_SIZE = 180
LEARNING_RATE = 1e-4
EPOCHS = 40
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CLASSIFIER_TRAINING = 'classifier_training_resnet18'
CKPT_SAVE_DIR = f"{CLASSIFIER_TRAINING}/classifier_ckpts"
PLOT_SAVE_DIR = f"{CLASSIFIER_TRAINING}/classifier_plots"
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png", dpi=150)
    plt.close()

def train_classifier():
    full_dataset = NpyDataset(root_dir=DATA_DIR, transform=classifier_transforms)
    
    dataset_size = len(full_dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = int(VAL_RATIO * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)
    model = torch.compile(model, mode='max-autotune')
    
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ], lr=1e-4)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting training on {DEVICE}...")
    
    epoch_pbar = tqdm(range(EPOCHS), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        batch_losses = []
        
        step_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", position=1, leave=False)
        
        for i, (inputs, labels) in enumerate(step_pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            optimizer.step()
            
            loss_val = loss.item()
            grad_norm_val = grad_norm.item()
            batch_losses.append(loss_val)
            running_loss += loss_val
            
            step_pbar.set_postfix({
                'Loss': f'{loss_val:.4f}',
                'Avg Loss': f'{np.mean(batch_losses):.4f}',
                'Grad Norm': f'{grad_norm_val:.3f}'
            })
        
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, DEVICE)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Train Acc': f'{train_acc:.1f}%',
            'Val Acc': f'{val_acc:.1f}%'
        })
        
        plot_metrics(train_losses, val_losses, train_accs, val_accs, PLOT_SAVE_DIR)
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            print(f"  ✓ New best val accuracy: {val_acc:.2f}%, saving checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, f"{CKPT_SAVE_DIR}/resnet18_best.pt")
        else:
            print(f"  Val accuracy: {val_acc:.2f}% (best: {best_val_acc:.2f}%)")
    
    print(f"\nTraining finished! Best model at epoch {best_epoch} with val accuracy: {best_val_acc:.2f}%")
    
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(f"{CKPT_SAVE_DIR}/resnet18_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    train_classifier()
