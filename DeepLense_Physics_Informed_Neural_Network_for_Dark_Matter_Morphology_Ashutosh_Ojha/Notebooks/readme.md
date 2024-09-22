# LensPIN Model Performance Comparison
## Architecture
![image](https://github.com/user-attachments/assets/75b0414a-bff4-4525-88b7-52f89511548c)

The LensPIN model was tested for two conditions:
1. **Small dataset** - 3000 samples per dataset
2. **Large dataset** - 18000 samples per dataset

## Large Dataset Results
The performance of different models on a large dataset is compared in the table below.

**Table 1: Performance Metrics of Different Models (Best in bold, Previous best underlined)**

| Model Name        | Parameters (Millions) | Micro F1 Score | No Subs. | CDM  | Axion |
|-------------------|-----------------------|----------------|---------------|----------|------|
| ResNet18          | 11.17                 | 0.910          | 0.98          | 0.90     | 0.92 |       
| ViT               | 13.7                  | 0.902          | 0.99          | 0.90     | 0.90 |       
| CaiT              | 13.7                  | 0.936          | 0.98          | 0.92     | 0.95 |       
| ViTSD             | 13.7                  | 0.911          | 0.99          | 0.91     | 0.90 |       
| Lensiformer       | 15.7                  | 0.976          | 1.00          | 0.97     | 0.98 |       
| LensCoAt_small    | 7.04                  | 0.994          | 1.00          | 0.99     | 0.99 |       
| LensCoAt_large    | 14.43                 | **0.999**      | **1.00**      | **1.00** | **1.00** |

## Small Dataset Results
For the small dataset case with 3000 images per class, the performance of the models was as follows.

**Table 2: Performance of Different Models (Parameters in millions; Best in bold, 2nd best underlined)**

| Model Name        | Parameters (Millions) | Accuracy | Micro F1 Score | No Subs. | CDM  | Axion |
|-------------------|-----------------------|----------|----------------|---------------|----------|------|
| ResNet18          | 11.17                 | 0.818    | 0.817          | 0.97          | 0.85     | 0.95 |       
| ViT               | 13.72                 | 0.863    | 0.864          | 0.99          | 0.66     | 0.96 |       
| CaiT              | 13.76                 | 0.878    | 0.871          | 0.99          | 0.64     | 0.96 |       
| ViTSD             | 13.73                 | 0.867    | 0.868          | 1.00          | 0.66     | 0.97 |       
| Lensformer        | 15.7                  | 0.957    | 0.959          | 1.00          | 0.99     | 0.99 |       
| LensPINN_small    | 7.17                  | 0.956    | 0.957          | 1.00          | 0.99     | 0.99 |       
| LensPINN_large    | 14.56                 | **0.996** | **0.996**      | **1.00**      | **1.00** | **1.00** |

