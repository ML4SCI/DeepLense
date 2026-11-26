# GSoC 2025 LensID

Gravitational Lens Finding and Classification project under DeepLense

Author: Dhruv Srivastava

## How to run the code

IPython notebooks work out-of-the-box (tested on Google Colab). If generating the dataloaders for the first time, uncomment the lines such as: (in vision_transformer_basic.ipynb)

```
# Create Datasets and Dataloaders
#train_dataset = MyDataset(train_dir)
#val_dataset = MyDataset(val_dir)
#dataset = MyDatasetViT(train_dir, vit_transforms)
#train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.75, 0.15, 0.1])

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

#print(f"Batch Size: {batch_size}")
#print(f"Number of Training Batches: {len(train_loader)}")
#print(f"Number of Validation Batches: {len(val_loader)}")

#Save the dataloader so that we don't have to bear with this pain again
#torch.save(train_loader, '/content/drive/MyDrive/Model_III_dataset/train_loader.pth')
#torch.save(val_loader, '/content/drive/MyDrive/Model_III_dataset/val_loader.pth')
```

Once dataloaders have been saved, simply load them through

```
#import data loaders from file
train_loader = torch.load('/content/drive/MyDrive/Model_III_dataset/train_loader.pth', weights_only=False)
val_loader = torch.load('/content/drive/MyDrive/Model_III_dataset/val_loader.pth', weights_only=False)
```

⚠️ **Note**

- The hard-coded Google Drive paths are intended for Google Colab usage.
- For local execution, update the dataset paths accordingly.
- For local execution, users are encouraged to replace Colab paths with:

```python
train_dir = "./data/Model_III"
```

and save dataloaders locally:

```
torch.save(train_loader, "./data/train_loader.pth")
```

### Dataset Structure (Model III):

The Vision Transformer models expect the dataset to be organized as follows:

```
data/
└── Model_III/
    ├── axion/
    │   ├── *.npy
    ├── cdm/
    │   ├── *.npy
    └── no_sub/
        ├── *.npy
```

Each .npy file contains a 2D array representing a grayscale lensing image.
The dataset is split into train/validation/test sets using
torch.utils.data.random_split inside the notebook.

### Local Setup (Non-Colab):

The project can be run locally without Google Colab.

Steps:

1. Clone the repository.

2. Create and activate a virtual environment.

3. Install required dependencies.

4. Place the dataset under data/Model_III/.

5. Open and run the relevant notebook (e.g. vision_transformer_basic.ipynb).

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy
```

### Notebook Execution Notes:

- Dataset and DataLoader creation cells should be run once.

- Saving DataLoaders using torch.save is optional and primarily intended for Colab usage.

- Training, validation, and testing splits are performed programmatically.
