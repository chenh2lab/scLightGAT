import torch
import numpy as np
import scanpy as sc
import optuna
import os
from scLightGAT.models.dvae_model import DVAE

# Optional: logger if needed
from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)

def extract_features_dvae(adata, model, device):
    """
    Extract latent features from trained DVAE model.
    """
    model.eval()
    with torch.no_grad():
        X = torch.tensor(adata.X.A if hasattr(adata.X, "A") else adata.X).float().to(device)
        mu, _ = model.encode(X)
        adata.obsm["X_dvae"] = mu.cpu().numpy()
    logger.info("Latent features extracted and stored in adata.obsm['X_dvae']")
    return adata

def optimize_dvae_params(train_func, trial):
    """
    Objective function for DVAE hyperparameter tuning.
    """
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    val_loss = train_func(hidden_dim, latent_dim, lr)
    return val_loss

def compare_hvg_features(adata, latent_key="X_dvae"):
    """
    Compare HVG-based and DVAE-based latent features via UMAP.
    """
    if "X_dvae" not in adata.obsm:
        raise ValueError("Latent features 'X_dvae' not found. Run DVAE first.")

    adata.obsm["X_hvg"] = adata.raw.X.toarray() if adata.raw is not None else adata.X
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["leiden"], title="DVAE Latent UMAP")

    sc.pp.neighbors(adata, use_rep="X_hvg")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["leiden"], title="HVG Feature UMAP")
