# scLightGAT ⚡
> A Self-Attention Graph Neural Network and LightGBM-based Tool for Cell Type Annotation in scRNA-seq

**scLightGAT** is an integrated tool for automated cell type annotation in **single-cell RNA sequencing (scRNA-seq)** datasets.  
It uniquely combines:

-  **Differential Gene Expression (DGE)** for prominent marker gene selection  
-  **Contrastive Denoising Variational Autoencoder (C-DVAE)** for latent feature extraction  
-  **LightGBM** for first-stage classification  
-  **Graph Attention Network (GAT)** for topology-aware prediction refinement

Tested on five independent GEO cancer datasets, scLightGAT achieves an average annotation accuracy of **0.962**, outperforming existing methods including CellTypist, CHETAH, and scGPT&#8203;:contentReference[oaicite:0]{index=0}.

---

## 📌 Key Features

- ✨ Feature fusion strategy: DGE-based marker genes + C-DVAE compressed HVGs
- ⚡ High-performance LightGBM classifier for fast, scalable cell prediction
- 🧠 Graph Attention Network for refining predictions with neighborhood context
- 🔬 CAF-aware design: includes rare stromal cell types often missed in other tools
- 🧰 CLI-ready, modular, and publication-friendly Python package

---



## Installation

```bash
git clone https://github.com/your-username/scLightGAT.git
cd scLightGAT
pip install -r requirements.txt
pip install -e .
```


---
## Quick Start
🏋️ Training
```bash
python scLightGAT/pipeline.py \
  --mode train \
  --train path/to/train.h5ad \
  --test path/to/test.h5ad \
  --output results/ \
  --model_dir results/models \
  --train_dvae \
  --use_gat \
  --dvae_epochs 20 \
  --gat_epochs 300
```
---

## Framework Description
scLightGAT is structured into three major stages:

1. Feature Extraction

- Removes low-quality cells, applies normalization and HVG filtering

- Performs DGE analysis and C-DVAE latent encoding on HVGs

2. Feature Fusion + Classification

- Concatenates DGE marker genes with C-DVAE latent vectors

- LightGBM predicts coarse cell types with high efficiency

3.Graph-based Refinement

- Constructs cell graphs using UMAP and Harmony representation

- GAT leverages attention across neighboring cells to refine predictions

This architecture improves cell type resolution (e.g. CAF subtypes) and maintains cross-dataset generalizability in cancer applications​

---

## Author
Alfie (Tsung-Hsien) Chuang
M.S. in Biomedical Electronics and Bioinformatics, National Taiwan University
GitHub: @tsunghsienchuang

---

## 📄 License
MIT License © 2025

