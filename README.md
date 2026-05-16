# Self-Supervised & Multi-Task Learning for Brain Tumor Segmentation and Classification

This repository contains a complete deep learning workflow implemented in PyTorch for brain tumor analysis on MRI scans. The framework employs a two-stage paradigm: **Self-Supervised Pre-training (SSL)** via random patch masking and pixel reconstruction, followed by **Multi-Task Learning (MTL)** fine-tuning for simultaneous tumor type classification and localization/segmentation.

It evaluates and directly compares two semantic segmentation backbones: a standard deep **U-Net** and a lightweight encoder-replaced **ResNet18-UNet**.

---

## 1. Project Overview
Medical imaging often suffers from limited annotations. This project leverages a self-supervised reconstruction task on unannotated/annotated pixel grids to warm up encoder representations before deploying downstream classifiers and segmentation decoders. 

The workflow transitions through two principal execution stages:
1. **Self-Supervised Pre-training (SSL):** Inputs are augmented and dynamically masked using a grid patch allocator. The core objective is to minimize the Mean Squared Error (MSE) over the masked patches to recover the original inputs.
2. **Multi-Task Learning Fine-Tuning (MTL):** Pre-trained weights are loaded into the models. The network is optimized simultaneously on two heads: a classification head identifying 4 paths (`gl`: Glioma, `me`: Meningioma, `pi`: Pituitary, `no`: No Tumor) and a segmentation decoder generating accurate target tumor contours.

---

## 2. Core Features
* **Dual Architecture Benchmarking:** Out-of-the-box configuration comparisons between a traditional high-capacity U-Net (~31M parameters) and an optimized ResNet18-UNet (~14.3M parameters) that significantly cuts down inference latency.
* **Dynamic Grid Patch Masking:** Built-in tensor processing functions (`apply_patch_mask`, `patch_normalize`) to dynamically drop patch components at adjustable masking ratios for SSL pre-training.
* **Robust Multi-Task Data Pipeline:** High-performance `SSLDataset` and `MTLDataset` routines equipped with dynamic augmentation.
* **Custom Compounded Loss Formulations:** * *Reconstruction Stage:* Mask-weighted Patch MSE Loss.
  * *Multi-Task Stage:* Combined BCE + Dice Coefficient loss (`localization_loss`) for fine-grained segmentation boundaries combined with Cross-Entropy Loss for multi-class classification.
* **Advanced Metrics & Analytics Reporting:** Computes macro metrics including Dice Coefficients, multi-class F1-scores, Hausdorff Distance (HD), and comprehensive validation composite scores.
* **Automated Visualizers:** Generates publication-ready figures for learning curves, comparative multi-model confusion matrices, and test-set prediction overlays.

---

## 3. Usage & Execution Guide

### Prerequisites
Ensure your localized Python environment is provisioned with core deep learning and computing packages:
```bash
pip install torch torchvision numpy pandas pillow matplotlib scikit-learn
