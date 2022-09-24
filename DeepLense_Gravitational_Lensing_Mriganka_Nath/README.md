# ML4SCI_Deeplense_Gravitational_Lensing

## Project Overview
Detecting strong lensing from images of space have multiple scientific applications like studying mass profiles of distant galaxies and clusters,
studying properties of galaxies at very high distances.It can also be used further in the DeepLense pipeline.
In this project we tried to apply differnt Machine learning algorithms to detect strong lensing.

## Files

|Files                   | Description                         |
|------------------------|-------------------------------------|
|Lensing_DomainAdaptation| Contains the all the code used      |
|Notebooks               | Tutoral & data exploration notebooks|



## Results

### Table showing AUCROC score for each Domain Adaptation technique(row) and the Encoder model used (column)

|                | Efficientnet_b2  | Resnet34  |  Densenet_121 |  ECNN  |
| -------------  | :---------------:  | :--------:  |  :------------: |  :-----: |
| ADDA           |        0.835     |   0.797   |     0.798     | `0.879`|
| Self-Ensembling|        0.604     |   0.445   |     0.537     |  0.449 |
| AdaMtch        |        0.824     |   0.8     |     0.763     |  0.795 |
             