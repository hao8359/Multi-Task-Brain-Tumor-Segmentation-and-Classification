# GP-8 – Self-Supervised Multi-Task Learning for Brain Tumor Segmentation and Classification

This repository contains the deep learning pipeline developed for **CM2026 Advanced Machine Learning in Healthcare** (Group 8). The code trains two multi-task models — one with a standard U-Net encoder and one with a ResNet18 backbone — using self-supervised pre-training via a Masked Autoencoder (MAE), followed by joint fine-tuning for brain tumor segmentation and classification from T1-weighted MRI.

**Group members:** Qusai Al Haj Ali, Mansour Arefi, De Chi Hao, Shreyas Balakarthikeyan

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Input Data](#input-data)
6. [Configuration](#configuration)
7. [Running the Notebook](#running-the-notebook)
8. [Model Architecture](#model-architecture)
9. [Training Pipeline](#training-pipeline)
10. [Evaluation](#evaluation)
11. [Outputs](#outputs)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

The pipeline processes T1-weighted MRI brain scans from the BRISC2025 dataset and performs the following steps automatically:

1. **Explore data** – parses filenames, verifies image and mask integrity, and plots the class distribution.
2. **Preprocess data** – computes z-score normalisation statistics, applies a stratified train/validation split, and constructs PyTorch datasets with augmentation.
3. **Stage 1 – SSL pre-training** – wraps each encoder in a Masked Autoencoder (MAE), masks 60% of image patches, and trains the encoder to reconstruct missing regions via MSE loss.
4. **Stage 2 – Multi-task fine-tuning** – loads the pre-trained encoder into the full model and fine-tunes jointly for segmentation (binary tumor mask) and classification (four tumor classes), using differential learning rates and early stopping.
5. **Evaluate** – loads the best checkpoint for each architecture and reports Dice, IoU, Hausdorff Distance, and macro F1-score on the held-out test set.
6. **Visualise** – generates learning curves, confusion matrices, side-by-side predictions, and an architecture cost comparison.

---

## Prerequisites

Before running the notebook, install the following software. All of it is free.

### 1 – Python (via Anaconda – recommended)

- Download Anaconda from: https://www.anaconda.com/download
- Choose the installer for your operating system and follow the on-screen instructions.
- After installation, open **Anaconda Navigator** or **Anaconda Prompt** (Windows) / **Terminal** (macOS and Linux).

### 2 – Kaggle (recommended execution environment)

The notebook is designed to run on **Kaggle** (free GPU tier). The data paths and working directory defaults assume the Kaggle filesystem. If running locally, adjust the `DATA_ROOT` and `WORK_DIR` constants accordingly (see [Configuration](#configuration)).

### 3 – Visual Studio Code (optional)

VS Code integrates well with Jupyter notebooks. Download it from: https://code.visualstudio.com/

Install the **Python** and **Jupyter** extensions from the Extensions panel.

---

## Installation

Follow these steps in order. Commands in code blocks are typed into your **Terminal** (macOS/Linux) or **Anaconda Prompt** (Windows).

### Step 1 – Clone the repository

```bash
cd path/to/your/chosen/directory
git clone <repository-url>
cd GP_8_SSL
```

If you do not have Git, download the ZIP from GitHub and unzip it.

### Step 2 – Create a Python environment

```bash
conda create -n gp8_ssl python=3.10
conda activate gp8_ssl
```

Run `conda activate gp8_ssl` at the start of every new terminal session.

### Step 3 – Install required packages

```bash
pip install -r requirements.txt
```

The required libraries are:

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework and pretrained model weights |
| `numpy` | Numerical array operations |
| `pandas` | Tabular data management and CSV export |
| `matplotlib` | Plotting and figure generation |
| `Pillow` | Image loading and resizing |
| `scikit-learn` | Train/validation splitting, metrics, and confusion matrices |
| `scipy` | Hausdorff distance computation |

### Step 4 – Install Jupyter

```bash
pip install notebook ipykernel
```

---

## Project Structure

```
GP_8_SSL/
│
├── GP_8_SSL.ipynb              ← Main notebook – start here
├── requirements.txt            ← List of required Python packages
├── README.md                   ← This file
│
├── data/
│   └── brisc2025/              ← BRISC2025 dataset root
│       ├── segmentation_task/  ← Paired image (.jpg) and mask (.png) files
│       └── classification_task/← Image files with class label encoded in filename
│
└── outputs/                    ← All results are saved here (created automatically)
    ├── best_ssl_encoder_unet.pth       ← Best U-Net SSL encoder checkpoint
    ├── best_ssl_encoder_resnet.pth     ← Best ResNet18 SSL encoder checkpoint
    ├── best_mtl_model_unet.pth         ← Best U-Net fine-tuned model checkpoint
    ├── best_mtl_model_resnet.pth       ← Best ResNet18 fine-tuned model checkpoint
    ├── zscore_stats.npy                ← Pre-computed normalisation statistics
    ├── train_split.csv                 ← Training split index
    ├── val_split.csv                   ← Validation split index
    ├── test_split.csv                  ← Test split index
    ├── learning_curves_compare.png     ← Training and validation curves
    ├── confusion_matrices_compare.png  ← Side-by-side confusion matrices
    └── test_predictions_compare.png    ← Example segmentation predictions
```

---

## Input Data

The pipeline expects the **BRISC2025** dataset placed under `DATA_ROOT`. The dataset is organised into two parallel folders: `segmentation_task` (paired image and mask) and `classification_task` (image with class label in filename). The segmentation folder is used as the primary data source since the class label can be parsed from the filename.

### Filename format

Filenames follow the pattern `brisc2025_train_00001_gl_ax_t1.jpg`, where the two-letter code encodes the tumor class:

| Code | Class |
|------|-------|
| `gl` | Glioma |
| `me` | Meningioma |
| `pi` | Pituitary tumor |
| `no` | No tumor |

### Segmentation masks

Masks are stored as `.png` files in `segmentation_task/` and are matched to images by filename (excluding extension). Binary masks are thresholded at pixel value 127.

### Sampling frequency

Images are resampled to `224 × 224` pixels. The pipeline accepts single-channel (grayscale) T1-weighted MRI only.

---

## Configuration

All tunable settings are defined at the top of **cell 2** of `GP_8_SSL.ipynb`. You do not need to edit any other cell unless changing the model architecture.

### Paths

```python
DATA_ROOT = "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025"
WORK_DIR  = "/kaggle/working"
```

Change these to local paths if not running on Kaggle.

### Image and class settings

```python
IMG_SIZE    = 224
N_CLASSES   = 4
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
```

### Stage 1 – SSL pre-training hyperparameters

```python
SSL_EPOCHS     = 100
SSL_LR         = 1e-4
SSL_BATCH_SIZE = 32
PATCH_SIZE     = 16
MASK_RATIO     = 0.60   # fraction of patches masked during MAE training
```

### Stage 2 – Multi-task fine-tuning hyperparameters

```python
MTL_EPOCHS     = 80
MTL_BATCH_SIZE = 16
ENC_LR         = 1e-5   # encoder (pre-trained) learning rate
DEC_LR         = 1e-4   # decoder and classifier learning rate
ALPHA          = 0.5    # weight between BCE and Dice in segmentation loss
W_SEG          = 2.0    # segmentation loss weight in multi-task objective
W_CLS          = 1.0    # classification loss weight in multi-task objective
PATIENCE       = 15     # early stopping patience (epochs)
```

### Architecture switches

```python
RUN_UNET   = True    # run the U-Net encoder experiment
RUN_RESNET = True    # run the ResNet18 encoder experiment
```

Set either to `False` to skip that architecture.

---

## Running the Notebook

### Option A – Kaggle (recommended for GPU access)

1. Upload `GP_8_SSL.ipynb` to a Kaggle notebook.
2. Attach the BRISC2025 dataset under **Add data**.
3. Enable GPU under **Session options → Accelerator**.
4. Click **Run All**.

### Option B – Visual Studio Code

1. Open VS Code and select **File → Open Folder**, then choose the project folder.
2. Open `GP_8_SSL.ipynb` in the Explorer panel.
3. Click **Select Kernel** (top-right) and choose the `gp8_ssl` conda environment.
4. Click **Run All** (▶▶) to execute the notebook from top to bottom.
5. To run a single cell, click inside it and press **Shift + Enter**.

### Option C – Jupyter in the browser

```bash
conda activate gp8_ssl
cd /path/to/GP_8_SSL
jupyter notebook
```

Open `GP_8_SSL.ipynb` in the browser and run all cells via **Cell → Run All**.

### Recommended execution order

Cells must be run in order from top to bottom. The notebook is structured as follows:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Introduction** | Project description and abstract (no code) |
| 2 | **Configuration and imports** | All hyperparameters, library imports, and device setup |
| 3 | **Data exploration** | Filename parser, dataset splits, class distribution plots |
| 4 | **Data preprocessing** | Z-score statistics, train/val split, datasets, dataloaders, MAE patch masking |
| 5 | **Model architecture** | All model class definitions (encoder, decoder, classification head, MAE wrapper) |
| 6 | **Model training** | Stage 1 SSL pre-training, then Stage 2 multi-task fine-tuning |
| 7 | **Model evaluation** | Load best checkpoints, evaluate on test set, side-by-side comparison table |
| 8 | **Final visualisations** | Learning curves, confusion matrices, prediction overlays, parameter counts |

---

## Model Architecture

Two encoder architectures are compared within the same multi-task pipeline. Both share the same decoder design, classification head, SSL pre-training procedure, and evaluation protocol.

### U-Net Encoder (baseline)

Five `DoubleConv → MaxPool` stages. Channel sizes progress as `64 → 128 → 256 → 512 → 1024`; spatial resolution decreases from `224 × 224` down to `14 × 14`. The bottleneck feature map has shape `(1024, 14, 14)`. Skip connections are taken at `224`, `112`, `56`, and `28` spatial resolutions.

### ResNet18 Encoder

A ResNet18 backbone with the first convolution layer adapted to accept single-channel input. Feature channel sizes progress as `64 → 64 → 128 → 256 → 512`; spatial resolution decreases from `112 × 112` down to `7 × 7`. The bottleneck feature map has shape `(512, 7, 7)`. Skip connections are taken at `112`, `56`, `28`, and `14` spatial resolutions.

### Shared Components

**Decoder** – U-Net-style upsampling path with transposed convolutions and skip connections matched to each encoder's spatial resolution schedule.

**Classification head** – Global average pooling collapses spatial dimensions to a single vector, followed by dropout and a fully connected layer producing four class logits.

**MAE wrapper** – A lightweight reconstruction decoder (discarded after pre-training) is attached to each encoder for Stage 1. Only the encoder weights transfer to Stage 2.

---

## Training Pipeline

### Stage 1 – Self-Supervised Pre-training (MAE)

60% of image patches (`patch_size = 16`) are randomly masked per image. The MAE is trained to reconstruct the missing patches using MSE loss computed only on masked locations. Per-patch normalisation is applied to the reconstruction target to reduce intensity variation. A cosine annealing learning rate schedule is used throughout. The best encoder checkpoint (lowest validation MSE) is saved to disk.

### Stage 2 – Multi-Task Fine-tuning

The pre-trained encoder is loaded into the full multi-task model. Differential learning rates are applied: `1e-5` for the encoder (to preserve learned SSL features) and `1e-4` for the decoder and classification head. The training objective is a weighted combination of segmentation and classification losses:

- **Segmentation loss** – α × BCE + (1 − α) × Dice, applied only to samples with a tumor mask (classes 0–2).
- **Classification loss** – Cross-entropy over all four classes.
- **Multi-task loss** – `w_seg × L_seg + w_cls × L_cls`.

Early stopping halts training if the validation composite metric does not improve for `PATIENCE` epochs. The best model checkpoint is saved based on the composite metric:

```
M_composite = 0.5 × Dice + 0.5 × Macro F1
```

---

## Evaluation

Both models are evaluated on the held-out test set using the following metrics:

| Metric | Description |
|--------|-------------|
| **Dice Score** | Overlap between predicted and ground-truth segmentation masks |
| **IoU** | Intersection over Union; penalises false positives more strictly than Dice |
| **Hausdorff Distance** | Maximum boundary distance between predicted and ground-truth masks (pixels) |
| **Macro F1-Score** | Classification performance averaged equally across all four classes |
| **Accuracy** | Overall classification accuracy |
| **Composite** | `0.5 × Dice + 0.5 × Macro F1` – used for model selection |

A side-by-side comparison table is printed with a Δ column (`ResNet18-Unet − U-Net`). A positive Δ indicates the ResNet variant outperformed the U-Net baseline on that metric (note: for Hausdorff Distance, a negative Δ is better).

---

## Outputs

After the notebook finishes, all results are saved under `WORK_DIR` (default: `/kaggle/working`):

### Model checkpoints

| File | Contents |
|------|----------|
| `best_ssl_encoder_unet.pth` | Best U-Net encoder weights from SSL pre-training |
| `best_ssl_encoder_resnet.pth` | Best ResNet18 encoder weights from SSL pre-training |
| `best_mtl_model_unet.pth` | Best U-Net multi-task model checkpoint (includes epoch and validation metrics) |
| `best_mtl_model_resnet.pth` | Best ResNet18 multi-task model checkpoint |
| `last_ssl_encoder_*.pth` | Final SSL encoder weights (last epoch) |
| `last_mtl_model_*.pth` | Final multi-task model weights (last epoch) |

### Data splits and statistics

| File | Contents |
|------|----------|
| `zscore_stats.npy` | Dataset mean and std used for z-score normalisation |
| `train_split.csv` | Image paths, mask paths, and labels for the training split |
| `val_split.csv` | Validation split |
| `test_split.csv` | Test split |

### Figures

| File | Contents |
|------|----------|
| `learning_curves_compare.png` | Training total loss, validation Dice, and validation composite for both models |
| `confusion_matrices_compare.png` | Row-normalised confusion matrices for U-Net and ResNet18-Unet side by side |
| `test_predictions_compare.png` | Input image, ground-truth mask, U-Net prediction, and ResNet18-Unet prediction for six test samples |

---

## Troubleshooting

**`FileNotFoundError: Cannot find segmentation/ folder under DATA_ROOT`**  
→ Check that `DATA_ROOT` points to the correct directory containing both `segmentation_task/` and `classification_task/` subfolders. Update the path in the configuration cell.

**`ModuleNotFoundError: No module named 'torch'`** (or any other package)  
→ Make sure the correct conda environment is active (`conda activate gp8_ssl`) and that `pip install -r requirements.txt` completed without errors.

**`No best checkpoint found`** during evaluation  
→ The training cell for that architecture must complete at least one epoch before the checkpoint is written. Check that `RUN_UNET` or `RUN_RESNET` is set to `True` and that training was not interrupted.

**CUDA out of memory**  
→ Reduce `SSL_BATCH_SIZE` or `MTL_BATCH_SIZE` in the configuration cell and re-run from the training cell. Setting `NUM_WORKERS = 0` can also reduce memory overhead on some systems.

**Hausdorff Distance returns `None` for some samples**  
→ This is expected when either the predicted mask or the ground-truth mask is entirely empty (no foreground pixels). These samples are skipped and counted in the `hd_skipped` field of the metrics dictionary.

**Plots do not appear in a plain terminal**  
→ Add `%matplotlib inline` as the first line of the imports cell and re-run the notebook from the top.

---

## Contact

For questions about this project, please contact the group members via the course repository or raise an issue on GitHub.
