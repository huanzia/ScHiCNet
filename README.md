# ScHiCNet: A Multi-Scale Feature-Guided Attention Framework for Enhancing Single-Cell Hi-C Data

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4.1-orange)](https://pytorch.org/)

## Overview

**ScHiCNet** is a deep learning framework designed to enhance the resolution and quality of single-cell Hi-C (scHi-C) contact matrices. By leveraging multi-scale feature extraction and attention-guided mechanisms, ScHiCNet effectively recovers chromatin interaction details from sparse and noisy scHi-C data.

It outperforms existing methods in terms of:
- **Structural Similarity (SSIM)**
- **Peak Signal-to-Noise Ratio (PSNR)**
- **Biological Reproducibility (HiCRep)**
- **Cross-species Generalization** (Robust performance across Human, Mouse, and Drosophila)

The architecture diagram of ScHiCNet is shown below:
![Model Architecture](https://github.com/user-attachments/assets/435bbde8-6482-4d05-84a0-b5bef20eec78)

---

## 🛠️ System Requirements

- **OS**: Linux (Recommended) or Windows
- **Python**: 3.8+
- **CUDA**: 11.x / 12.x (Strongly recommended for GPU acceleration)

### Core Dependencies
- PyTorch >= 2.0.0
- NumPy >= 1.23.5
- SciPy, Pandas, Matplotlib
- Cooler, h5py
- PyTorch Lightning

---

## 🚀 Installation

You can set up the environment using `conda` (recommended) or `pip`.

### Option 1: Using Conda (Recommended)
```bash
# 1. Clone the repository
git clone [https://github.com/huanzia/ScHiCNet.git](https://github.com/huanzia/ScHiCNet.git)
cd ScHiCNet

# 2. Create environment from the provided YAML file
conda env create -f schicnet_cu126.yml

# 3. Activate the environment
conda activate schicnet_cu126
```

📂 Project Structure

The directory structure of ScHiCNet is organized as follows:
ScHiCNet/
├── Models/       # Source code for ScHiCNet and baseline models (e.g., ScHiCAtt, DeepHiC)
├── ProcessData/  # Scripts for data preprocessing and tensor generation
├── Training/     # Main training scripts (entry point: schicnet_train.py)
├── Utils/        # Utility functions for metrics (SSIM, GenomeDISCO), loss, and I/O
├── experi/       # Configuration scripts for batch experiments
└── Output/       # Scripts for visualization and result extraction

## 📖 Data Preparation

ScHiCNet accepts Hi-C data in **`.mcool`** format (multiresolution cooler files).

### Public Datasets
The datasets used in our paper can be downloaded from:
- **Drosophila**: [GSE131811](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131811)
- **Human**: [GSE130711](https://salkinstitute.app.box.com/s/fp63a4j36m5k255dhje3zcj5kfuzkyj1)
- **Mouse**: [GSE162511](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162511)


## Running ScHiCNet

```bash
Step 1: Navigate to the project directory.
cd ./Training
Step 2: Training your data.
python schicnet_train.py
```
**Arguments:**

* `-b` (`--batch_size`): Batch size for training (e.g., 16).
* `-l` (`--cell_line`): Name of the cell line (e.g., Mouse,Human,Dros).
* `-n` (`--cell_number`): The specific cell index or ID to process(e.g.,1).
* `-e` (`--epochs`): Total number of training epochs(e.g.,300).
* `-p` (`--percent`): The sampling ratio(e.g.,0.75,0.45,0.1).
