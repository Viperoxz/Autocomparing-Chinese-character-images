# Siamese Network Project

## Overview
This project implements two Siamese Network models for comparing handwritten character images (64x64, grayscale):
- SiameseNetwork (custom CNN)
- ResNetSiameseNetwork (based on ResNet18)

## Directory Structure
- `data/`: Contains HDF5 datasets (You can get the HDF5 datasets at [this link](https://drive.google.com/drive/folders/1kTcnoU773tdvbDB94knD6g8Ien8qMhHf)
) and test samples.
- `models/`: Contains model definitions and checkpoints.
- `scripts/`: Contains training and testing scripts.
- `utils/`: Contains utility functions (dataset, loss, helpers).
- `outputs/`: Contains logs and plots.  