import torch
import os
import pickle 
import torch.nn as nn
import numpy as np
from torchvision import models

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from models.unet_sa import UNet_conditional, UNet_linear_conditional
from models.ddpm import Diffusion

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./conditional_ddpm_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()

model_ft = models.resnet18(pretrained=False)
model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 3))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model_2_path = "saved_models/ResNet18_Model2.pth"
model_ft = torch.load(resnet_model_2_path)#, map_location=device)
model = model_ft.to(device)
model.eval()

# Load model
model_diffusion = UNet_linear_conditional(config)
model_diffusion.load_state_dict(torch.load('saved_models/epochs_1000_conditional_ckpt_model2.pt'))#, map_location=torch.device('cpu'))
model_diffusion = model_diffusion.to(device=config.device)

# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)

# Roc curve
num_classes = 3
all_labels = []
all_scores = []

number_of_sample = 100
with torch.no_grad():
    labels_nosub = torch.full([number_of_sample,], 2, dtype=torch.long).to(device)
    labels_axion = torch.zeros([number_of_sample,], dtype=torch.long).to(device)
    labels_cdm   = torch.ones([number_of_sample], dtype=torch.long).to(device)

    all_labels.extend(labels_nosub.cpu().numpy())
    all_labels.extend(labels_axion.cpu().numpy())
    all_labels.extend(labels_cdm.cpu().numpy())


    gen_nosub = diffusion.sample_conditional(model_diffusion, number_of_sample, labels_nosub)
    gen_axion = diffusion.sample_conditional(model_diffusion, number_of_sample, labels_axion)
    gen_cdm = diffusion.sample_conditional(model_diffusion, number_of_sample, labels_cdm)

    gen_nosub = model(gen_nosub)
    gen_axion = model(gen_axion)
    gen_cdm   = model(gen_cdm)

    probabilities_nosub = torch.nn.functional.softmax(gen_nosub, dim=1)
    probabilities_axion = torch.nn.functional.softmax(gen_axion, dim=1)
    probabilities_cdm   = torch.nn.functional.softmax(gen_cdm, dim=1)
    
    
    all_scores.extend(probabilities_cdm.cpu().numpy())
    all_scores.extend(probabilities_nosub.cpu().numpy())
    all_scores.extend(probabilities_axion.cpu().numpy())


# Binarize the labels
#print(all_labels)
#print(all_scores)
y_true = label_binarize(all_labels, classes=list(range(num_classes)))

# Ensure y_true and all_scores are numpy arrays
y_true = np.array(y_true)
all_scores = np.array(all_scores)
#print(y_true)
#print(all_scores)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], all_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), all_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure(figsize=(10, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"], color='darkorange', linestyle='--', lw=2, label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass Classification')
plt.legend(loc="lower right")
plt.savefig(os.path.join("plots", f"roc_1000.jpg"))