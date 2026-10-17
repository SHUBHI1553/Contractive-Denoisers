# Provably Contractive and High-Quality Denoisers for Convergent Restoration

**Official PyTorch Implementation | BMVC 2026 (Oral Presentation)**

**Authors:** Shubhi Shukla and Pravin Nair

**Conference:** British Machine Vision Conference (BMVC), 2026

## Overview

This repository contains the official implementation of our paper, **"Provably Contractive and High-Quality Denoisers for Convergent Restoration,"** accepted for an **oral presentation at BMVC 2026**.

We propose a contractive denoising architecture designed for high-quality image restoration with convergence guarantees. The proposed denoiser is integrated into a Plug-and-Play (PnP) framework for image deblurring and super-resolution.

The repository provides pretrained model weights and evaluation scripts to reproduce the denoising, deblurring, and super-resolution results presented in the paper. The training implementation is also provided to support reproducibility and further research.

## Repository Structure

```text
Contractive-Denoisers/
├── train.py                  # Training script
├── denoising_test.py         # Image denoising evaluation
├── pnp_deblurring.py         # PnP image deblurring
├── pnp_superresolution.py    # PnP image super-resolution
├── model.py                  # Model architecture
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── images/                   # Input images for evaluation
├── models/                   # Pretrained model weights
└── outputs/                  # Generated restoration results
```

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/SHUBHI553/Contractive-Denoisers.git
cd Contractive-Denoisers
pip install -r requirements.txt
```

## Datasets

### Training Dataset

The proposed denoiser is trained using **400 images from the BSD500 dataset**, consisting of:

- 200 images from the original training split.
- 200 images from the original testing split.

The dataset can be downloaded from the [official BSD500 dataset page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).

### Testing Datasets

The following benchmark datasets are used to evaluate the denoising performance reported in the paper.

| Task | Testing Datasets |
|---|---|
| RGB image denoising | CBSD68, Kodak24, Urban100 |
| Grayscale image denoising | BSD68, Set12, Urban100 |

These datasets are used for quantitative evaluation and comparison with existing image denoising methods.

### Dataset Preparation

The datasets are not included in this repository and must be downloaded separately.

For training, download BSD500 and combine the 200 training images and 200 testing images into a single directory:

```text
data/
└── BSD500/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ...
```

The training script reads the images from this directory and automatically extracts patches. No separate patch-generation script is required.

The default data preparation settings are:

| Parameter | Value |
|---|---|
| Training images | 400 (200 train + 200 test) |
| Patch size | 64 × 64 |
| Patch stride | 4 |
| Training/validation split | 90% / 10% |
| Data augmentation | Random horizontal and vertical flips, and rotations |

The default dataset directory is `./data/BSD500`. This path can be modified in `train.py`.

## Reproducing the Paper Results

We provide pretrained model weights and evaluation scripts to reproduce the image restoration experiments presented in the paper.

The pretrained weights are available in `models/`, and sample images are provided in `images/`.

Run the following commands from the repository root.

### Image Denoising

```bash
python denoising_test.py
```

### Image Deblurring

```bash
python pnp_deblurring.py
```

### Image Super-Resolution

```bash
python pnp_superresolution.py
```

The evaluation scripts use the pretrained denoiser to reproduce the corresponding restoration results. Ensure that the model and input image paths in the scripts are configured correctly.

## Training

The training script is provided for training the proposed denoiser from scratch.

### Training Configuration

| Parameter | Value |
|---|---|
| Noise model | Additive white Gaussian noise |
| Noise standard deviation | 15/255 |
| Batch size | 256 |
| Number of epochs | 50 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Loss function | Weighted combination of L1 and SSIM losses |

### Run Training

```bash
python train.py
```

Model checkpoints are saved after each epoch in:

```text
outputs/checkpoints/
```

## Outputs

The evaluation scripts generate restored images for the corresponding image restoration tasks.

The `outputs/` directory is used for generated restoration results, while training checkpoints are stored in `outputs/checkpoints/`.

## Citation

If you find this work useful in your research, please cite:

**Shubhi Shukla and Pravin Nair.**  
*Provably Contractive and High-Quality Denoisers for Convergent Restoration.*  
British Machine Vision Conference (BMVC), 2026. **Oral Presentation.**

## Contact

For questions regarding the implementation or reproduction of the experimental results, please open an issue in this repository.tputs/ directory.
