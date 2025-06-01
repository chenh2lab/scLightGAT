# scLightGAT

A Novel Multi-Stage Framework Integrating C-DVAE, LightGBM, and GATs for Single-Cell RNA-Seq Cell-Type Annotation

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Installation](#installation)  
4. [Data Requirements & Preprocessing](#data-requirements--preprocessing)  
5. [Pipeline Architecture](#pipeline-architecture)  
   1. [Stage (a): Feature Compression via C-DVAE](#stage-a-feature-compression-via-c-dvae)  
   2. [Stage (b): Initial Cell-Type Estimation with LightGBM](#stage-b-initial-cell-type-estimation-with-lightgbm)  
   3. [Stage (c): Graph Attention Network for Label Refinement](#stage-c-graph-attention-network-for-label-refinement)  
6. [Usage](#usage)  
7. [Configuration & Hyperparameters](#configuration--hyperparameters)  
8. [Performance & Benchmarks](#performance--benchmarks)  
9. [Directory Structure](#directory-structure)  
10. [Citation](#citation)  
11. [License](#license)  

---

## Project Overview

scLightGAT is a Python-native, biologically informed, and computationally scalable framework designed to tackle the challenges of high-dimensional, noisy single-cell RNA-Seq (scRNA-seq) data. By combining Contrastive Denoising Variational Autoencoder (C-DVAE), LightGBM, and Graph Attention Networks (GATs) in a three-stage pipeline, scLightGAT achieves robust and accurate cell-type annotation across diverse datasets—achieving an average accuracy of **0.962** on five independent GEO test sets.

- **Three-Stage Design**  
  1. **Feature Compression**: A C-DVAE compresses highly variable genes (HVGs) into a compact latent space (Z), while differential gene expression (DGE) analysis selects biologically discriminative markers (M_DGE).  
  2. **Initial Classification**: A LightGBM classifier is trained on fused features \[Z ∥ M_DGE\], producing preliminary cell-type probability scores (\(\hat y_\text{LGBM}\)).  
  3. **Label Refinement**: A two-layer GAT module refines LightGBM’s predictions by modeling neighborhood relationships on a single-cell graph (SCG).  

- **Biological Validation**  
  scLightGAT excels at annotating both immune and non-immune compartments, with particular strength in resolving challenging subtypes such as cancer-associated fibroblasts (CAFs). When benchmarked on a hold-out CAF dataset, scLightGAT outperformed existing methods (e.g., CellTypist), achieving an accuracy of **0.9124 ± 0.0045**.

- **Scalability & Efficiency**  
  Despite integrating deep learning (C-DVAE + GAT), scLightGAT maintains competitive inference times—averaging **20.2 minutes** per dataset on an NVIDIA RTX A5000 GPU—by leveraging LightGBM’s leaf-wise gradient boosting for the bulk classification step.

---

## Key Features

- **Contrastive Latent Embedding**  
  - C-DVAE encoder compresses HVG set (\(M_\text{HVG} ∈ ℝ^{n×d_1}\)) into a \(300\)-dimensional latent matrix \(Z ∈ ℝ^{n×k}\).  
  - InfoNCE loss enforces cluster cohesion for same-type cells and separation for different types.  

- **Prominent Gene Selection**  
  - Differential gene expression (DGE) analysis selects top markers (\(M_\text{DGE} ∈ ℝ^{n×d_2}\)).  
  - Fused feature matrix \(F = [\,Z ∥ M_\text{DGE}\,] ∈ ℝ^{n×(k + d_2)}\).  

- **LightGBM Classifier**  
  - Leaf-wise gradient boosting on fused features.  
  - Produces probability vector \(\hat y_\text{LGBM} ∈ ℝ^{n×C}\) (where \(C\) = number of cell types).  

- **Graph Attention Refinement (GATs)**  
  - Constructs a single-cell graph (SCG) with:  
    1. Harmony-corrected embeddings  
    2. DGE features  
    3. UMAP coordinates  
    4. LightGBM logits (\(\hat y_\text{LGBM}\))  
    5. Leiden cluster labels  
  - Two-layer GAT:  
    - **Layer 1**: Multi-head attention (\(8\) heads) → ELU activation → dropout.  
    - **Layer 2**: Single-head attention → residual connections → log-softmax.  
  - Final refined labels assigned via \(\arg\max\) on GAT output probabilities.  

---

## Installation

```bash
# 1. Clone or download this repository
git clone https://github.com/chenh2lab/CTH_2023.git
cd CTH_2023

# 2. Create and activate a Python 3.8+ virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt
