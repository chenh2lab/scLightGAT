import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
import optuna
from models.dvae_model import DVAEModel
import gc
import logging
from sklearn.metrics import silhouette_score
from typing import List, Optional, Dict, Tuple, Any


# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_silhouette_score(latent_features: np.ndarray, labels: np.ndarray) -> float:
    """计算 Silhouette Score 用于评估 cluster"""
    score = silhouette_score(latent_features, labels)
    return score


def calculate_cluster_loss(z: torch.Tensor, 
                         labels: torch.Tensor, 
                         temperature: float = 0.5) -> torch.Tensor:
    """Calculate clustering loss within batch"""
    z = F.normalize(z, dim=1)
    
    
    similarity_matrix = torch.matmul(z, z.T)  # [batch_size, batch_size]
    
    
    labels = labels[:z.size(0)] 
    labels = labels.view(-1, 1)
    
    # 創建mask
    mask = torch.eq(labels, labels.T).float()  # [batch_size, batch_size]
    mask = mask.fill_diagonal_(0)
    
    
    pos_pairs = mask * similarity_matrix
    pos_mask = (mask > 0)
    if pos_mask.sum() > 0: 
        pos_mean = pos_pairs[pos_mask].mean()
    else:
        return torch.tensor(0., device=z.device)  
        
    # 計算負樣本對
    neg_mask = 1 - mask
    neg_pairs = neg_mask * similarity_matrix
    neg_pairs = neg_pairs[neg_mask.bool()].mean()
    
    # 計算loss
    loss = -torch.log(
        torch.exp(pos_mean / temperature) / 
        (torch.exp(pos_mean / temperature) + 
         torch.exp(neg_pairs / temperature) + 1e-6)
    )
    
    return loss

def create_umap_visualization(latent_features: np.ndarray,
                            cell_types: np.ndarray, 
                            title: str,
                            save_path: str = None):
    """Create and save UMAP visualization"""
    # Create temporary AnnData
    temp_adata = sc.AnnData(latent_features)
    temp_adata.obs['cell_type'] = cell_types
    
    # Compute UMAP
    sc.pp.neighbors(temp_adata, n_neighbors=10, use_rep='X')
    sc.tl.umap(temp_adata, min_dist=0.1)
    
    silhouette = calculate_silhouette_score(latent_features, cell_types)
    
    # Plot
    plt.figure(figsize=(12, 10))
    ax = sc.pl.umap(
        temp_adata,
        color='cell_type',
        title=title,
        frameon=False,
        legend_loc='right margin',
        legend_fontsize=8,
        size=8,
        alpha=0.7,
        show=False,
        return_fig=True
    ).axes[0]
    
    
    ax.text(
        1.02, 0.98, 
        f"Silhouette Score: {silhouette:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def extract_features_dvae(adata: sc.AnnData,
                         params: Dict[str, Any] = None,
                         use_hvg: bool = True,
                         save_path: str = None) -> Dict:
    """Extract features using DVAE with optional HVG selection"""
    if params is None:
        params = {
            'latent_dim': 300,
            'cluster_weight': 1,
            'temperature': 0.01,
            'epochs': 20,
            'batch_size': 128,
            'balanced_count': 10000,
            'n_hvgs': 3000,
            'learning_rate': 1e-5,
            'kl_weight': 0.03
        }
    
    # 準備數據
    if "log_transformed" not in adata.layers:
        adata.layers["log_transformed"] = np.log1p(adata.X)
    
    # 選擇特徵
    if use_hvg:
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=params['n_hvgs'],
            batch_key='Celltype_training',
            subset=True,
            layer="log_transformed"
        )
        matrix = adata[:, adata.var.highly_variable].to_df(layer="log_transformed")
        feature_names = adata.var_names[adata.var.highly_variable].tolist()
    else:
        matrix = adata.to_df(layer="log_transformed")
        feature_names = adata.var_names.tolist()

    # 準備標籤
    encoder = LabelEncoder()
    cell_type_encoded = encoder.fit_transform(adata.obs['Celltype_training'])
    cell_types = adata.obs['Celltype_training'].values

    # 準備訓練數據
    X_train = matrix.values
    y_train = cell_type_encoded
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = DVAEModel(
        input_dim=X_train.shape[1],
        latent_dim=params['latent_dim'],
        cluster_weight=params['cluster_weight'],
        temperature=params['temperature'],
        noise_factor=0.1
    ).to(device)
    
    # 優化器設置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=params['epochs'],
        eta_min=1e-6
    )
    
    # 準備訓練資料
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    # 訓練紀錄
    train_losses = []
    best_loss = float('inf')
    best_model = None
    no_improve = 0
    patience = 15

    # 訓練循環
    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(loader):
            batch = batch[0].to(device)
            optimizer.zero_grad()
            
            x_hat, mean, log_var, z = model(batch)
            
            # 取對應的batch標籤
            start_idx = batch_idx * params['batch_size']
            end_idx = min((batch_idx + 1) * params['batch_size'], len(y_train))
            batch_labels = torch.tensor(y_train[start_idx:end_idx], device=device)
            
            reconstruction_loss = F.mse_loss(x_hat, batch)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            
            if epoch > 10:
                cluster_loss = calculate_cluster_loss(
                    z, 
                    batch_labels,
                    temperature=params['temperature']
                )
                loss = (reconstruction_loss + 
                       params['kl_weight'] * kl_divergence + 
                       params['cluster_weight'] * cluster_loss)
            else:
                loss = reconstruction_loss + params['kl_weight'] * kl_divergence
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 清理記憶體
            del x_hat, mean, log_var, z, loss
            torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        
        # 更新學習率
        scheduler.step()
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{params['epochs']}], "
                       f"Loss: {avg_loss:.4f}, "
                       f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # 載入最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # 生成latent features
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for i in range(0, len(X_train), params['batch_size']):
            batch = torch.tensor(
                X_train[i:i + params['batch_size']], 
                dtype=torch.float32
            ).to(device)
            _, _, _, z = model(batch)
            latent_features.append(z.cpu().numpy())
    
    latent_features = np.concatenate(latent_features, axis=0)
    
    # 創建UMAP可視化
    title = f"DVAE Latent Space ({'with' if use_hvg else 'without'} HVG)"
    create_umap_visualization(
        latent_features=latent_features,
        cell_types=cell_types,
        title=title,
        save_path=save_path if save_path else None
    )
    
    return {
        'model': model,
        'latent_features': latent_features,
        'losses': train_losses,
        'final_loss': train_losses[-1],
        'feature_names': feature_names
    }
def compare_hvg_effect(adata: sc.AnnData, params: Dict[str, Any] = None) -> Dict:
    
    results = {}
    
    
    logger.info("Training DVAE with HVG selection...")
    results['with_hvg'] = extract_features_dvae(
        adata,
        params=params,
        use_hvg=True,
        save_path='dvae_with_hvg.png'
    )
    
    
    logger.info("Training DVAE without HVG selection...")
    results['without_hvg'] = extract_features_dvae(
        adata,
        params=params,
        use_hvg=False,
        save_path='dvae_without_hvg.png'
    )
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(results['with_hvg']['losses'], label='With HVG')
    plt.plot(results['without_hvg']['losses'], label='Without HVG')
    plt.title('DVAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('dvae_training_loss.png')
    plt.show()
    
    logger.info(f"Final loss with HVG: {results['with_hvg']['final_loss']:.4f}")
    logger.info(f"Final loss without HVG: {results['without_hvg']['final_loss']:.4f}")
    
    return results


def optimize_dvae_params(adata: sc.AnnData, 
                        n_trials: int = 20,
                        use_hvg: bool = True) -> Dict:
    """使用Optuna優化DVAE參數"""
    def objective(trial):
        params = {
            'latent_dim': trial.suggest_int('latent_dim', 64, 256),
            'cluster_weight': trial.suggest_float('cluster_weight', 0.1, 2.0),
            'temperature': trial.suggest_float('temperature', 0.01, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'kl_weight': trial.suggest_float('kl_weight', 0.01, 0.1),
            'epochs': 100,  # 固定參數
            'batch_size': 256,
            'balanced_count': 10000,
            'n_hvgs': 4000 if use_hvg else None
        }
        
        try:
            results = extract_features_dvae(
                adata,
                params=params,
                use_hvg=use_hvg
            )
            return -results['final_loss']  # 最小化損失
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return float('inf')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Best parameters found:")
    for key, value in study.best_params.items():
        logger.info(f"{key}: {value}")
    
    return study.best_params

# 使用範例
if __name__ == "__main__":
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    """
    # 1. 單純提取特徵
    results = extract_features_dvae(
        adata,
        params=None,  # 使用預設參數
        use_hvg=True,
        save_path='dvae_features.png'
    )
    
    # 2. 比較HVG效果
    hvg_comparison = compare_hvg_effect(adata)
    
    # 3. 優化參數
    best_params = optimize_dvae_params(
        adata,
        n_trials=20,
        use_hvg=True
    )
    
    # 4. 使用最佳參數重新訓練
    final_results = extract_features_dvae(
        adata,
        params=best_params,
        use_hvg=True,
        save_path='dvae_optimized.png'
    )
    """