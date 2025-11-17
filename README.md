# ScHiCNet ScHiCNet: A Multi-Scale Feature-Guided Attention Framework for Enhancing Single-Cell Hi-C Data
ScHiCNet is a deep learning model designed to enhance the resolution and accuracy of single-cell Hi-C data for studying chromatin interactions.

## Summary


ScHiCNet is a deep learning model designed to enhance single-cell Hi-C (scHi-C) data by improving its resolution and accuracy. It utilizes multi-scale convolutions and attention mechanisms to recover chromatin interaction maps, outperforming existing methods in structural similarity and biological reproducibility across species like human, mouse, and Drosophila. ScHiCNet is a powerful tool for advancing genomic research on chromatin organization.
<img width="889" height="670" alt="arh" src="https://github.com/user-attachments/assets/435bbde8-6482-4d05-84a0-b5bef20eec78" />

## Dependency
ScHiCNet is written in Python3 with PyTorch framework. It demands Python version 3.8+
Other python packages used in this repo (version numbers are recommended):

-pytorch 2.4.1
-pytorch-lightning 1.0.3
-torchvision 0.19.1
-numpy 1.23.5
-scipy 1.5.2
-pandas 1.1.3
-scikit-learn 1.3.2
-h5py 3.11.0
-cooler 0.8.11
-pyfaidx 0.8.1.3
-pypairix 0.3.9
-networkx 3.1
-matplotlib 3.3.2
-tensorboard 2.14.0
-tqdm 4.51.0
-pyyaml 6.0.2

> Note: GPU acceleration (CUDA 12) is strongly recommended.
