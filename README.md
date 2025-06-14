# scLightGAT: A LightGBM-based Framework Integrating C-DVAE and GATs for Robust Single-cell RNA-seq Cell Type Annotation

## Overview
We propose **scLightGAT**, a Python-native, biologically informed, and computationally scalable framework for cell-type annotation. scLightGAT combines machine learning and deep learning techniques through a three-stage architecture:
1. **C-DVAE**: Contrastive Denoising Variational Autoencoder extracts low-dimensional latent features from highly variable genes (HVGs).
2. **LightGBM**: A gradient-boosted classifier uses the fused latent (Z) and DGE marker (M_DGE) features for an initial cell-type prediction.
3. **GATs**: Graph Attention Networks refine LightGBM’s output by modeling neighborhood interactions on a single-cell graph (SCG).

Designed for high-quality, well-preprocessed datasets, scLightGAT achieves accurate annotations across both immune and non-immune compartments and demonstrates particular strength in resolving complex subtypes like cancer-associated fibroblasts (CAFs). Benchmarking across five public Gene Expression Omnibus (GEO) datasets confirms its ability to balance annotation accuracy, runtime efficiency, and subtype resolution—making it a practical tool for large-scale scRNA-seq studies.

---

## Installation

```bash
git clone https://github.com/chenh2lab/scLightGAT.git
cd scLightGAT
pip install -r requirements.txt
pip install -e .
```

---



## Pipeline Overview

![Pipeline Overview](docs/figures/overall.png)

**Figure 1.** Overview of the scLightGAT framework.  
**(a)** *Feature compression* via C-DVAE: highly variable genes (Mₕᵥgₛ) are selected and combined with DGE marker genes (M_DGE), then passed through a Contrastive Denoising VAE to produce low-dimensional latent embeddings Z; Z and M_DGE are concatenated to form the fused feature matrix F.  
**(b)** *Initial cell-type estimation* with LightGBM: the fused features F are fed into a gradient-boosted decision tree classifier, yielding per-cell logits and class probabilities from which initial predictions ŷ_LGBM are obtained via argmax.  
**(c)** *Prediction refinement* via a two-layer Graph Attention Network: a single-cell graph (SCG) built from Harmony embeddings, UMAP coordinates, and Leiden labels is input to two GATConv layers to refine ŷ_LGBM and produce the final labels ŷ_GAT.

---

## Quick Start

### 🏋️ Training

```python
import sys, importlib
import scLightGAT.pipeline

# Reload to pick up any local changes
importlib.reload(scLightGAT.pipeline)

# Simulate command-line arguments for training
sys.argv = [
    "pipeline.py",
    "--mode", "train",
    "--train", "/home/dslab_cth/scLightGAT/data/train.h5ad",
    "--test", "/home/dslab_cth/scLightGAT/data/GSE153935.h5ad",
    "--output", "/home/dslab_cth/scLightGAT/results",
    "--model_dir", "/home/dslab_cth/scLightGAT/scLightGAT/models",
    "--train_dvae",
    "--use_gat",
    "--dvae_epochs", "5",
    "--gat_epochs", "300"
]

from scLightGAT.pipeline import main
main()
```

- `--mode train`
- `--train <path/to/train.h5ad>`
- `--test  <path/to/test.h5ad>`
- `--output <results_dir>`
- `--model_dir <model_save_dir>`
- `--train_dvae` (flag to train C-DVAE)
- `--use_gat` (flag to enable GAT refinement)
- `--dvae_epochs <int>` (number of C-DVAE epochs)
- `--gat_epochs <int>` (number of GAT epochs)

After training, the results folder will contain:
- Trained C-DVAE encoder/decoder weights
- LightGBM model & logits
- GAT model weights & refined predictions
- UMAP plots and confusion matrices (if enabled)

---
## Experimental Results

- [Cross-validation ground truth](exp_results/Cross_validation(PanlaoDB)_groundtruth/)
- [Existing methods comparisons](exp_results/Existing_methods_comparisons/)
- [Harmony‐corrected UMAPs](exp_results/Harmony_corrected_umap/)
- [Doublets information](exp_results/doublets_info/)
- [Leiden clusters](exp_results/leiden_clusters/)

---

## Authors

**Tsung-Hsien Chuang**, **Cheng-Yu Li**, **Liang-Chuan Lai**, **Tzu-Pin Lu**, **Mong-Hsun Tsai**, **Eric Y. Chuang***, and **Hsiang-Han Chen***
- Tsung-Hsien Chuang, Liang-Chuan Lai, Tzu-Pin Lu, Mong-Hsun Tsai, Eric Y. Chuang are with the National Taiwan University, Taipei, Taiwan.
- Cheng-Yu Li and Hsiang-Han Chen are with the National Taiwan Normal University, Taipei, Taiwan.
- *Correspondence to: Hsiang-Han Chen (chenh2@ntnu.edu.tw) and Eric Y. Chuang (chuangey@ntu.edu.tw).

---

## Acknowledgments

This work was supported in part by the National Science and Technology Council (Taiwan) under grant NSTC 113-2222-E-003-001.

---

© 2025 Tsung-Hsien Chuang, Cheng-Yu Li, Liang-Chuan Lai, Tzu-Pin Lu, Mong-Hsun Tsai, Eric Y. Chuang, and Hsiang-Han Chen. All rights reserved.
