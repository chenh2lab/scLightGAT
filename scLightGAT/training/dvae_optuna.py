import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from datetime import datetime
import torch
import optuna

from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)
BASE_DIR = "/home/dslab_cth/"  
sys.path.append(BASE_DIR)


from models.dvae_model import DVAEModel  # 確保dvae_model.py在正確的路徑下
from Study.dvae_evaluation import compare_balanced_vs_imbalanced, optimize_dvae_params, save_results

# 設置日誌
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'dvae_training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    # 設置參數
    params = {
        'data_path': os.path.join(BASE_DIR, 'train.h5ad'),  # 替換為你的數據路徑
        'result_dir': os.path.join(BASE_DIR, '/Study/results'),
        'log_dir': os.path.join(BASE_DIR, 'logs'),
        'n_trials': 1,  # Optuna優化的試驗次數
        'base_params': {
            'epochs': 1,
            'batch_size': 64,
            'balanced_count': 10000,
            'n_hvgs': 3000
        }
    }
    

    logger = setup_logging(params['log_dir'])
    logger.info("Starting DVAE training script")
    
    try:
        # 創建結果目錄
        os.makedirs(params['result_dir'], exist_ok=True)
        
        # 記錄系統信息
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # 加載數據
        logger.info(f"Loading data from {params['data_path']}")
        adata = sc.read_h5ad(params['data_path'])
        logger.info(f"Data shape: {adata.shape}")
        
        # 優化參數
        logger.info("Starting parameter optimization")
        best_params = optimize_dvae_params(
            adata,
            n_trials=params['n_trials'],
            base_params=params['base_params']
        )
        
        # 使用最佳參數進行訓練
        logger.info("Starting final training with best parameters")
        training_params = {**params['base_params'], **best_params}
        results = compare_balanced_vs_imbalanced(adata, **training_params)
        
        # 保存結果
        logger.info("Saving results")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(params['result_dir'], f'run_{timestamp}')
        save_results(results, save_dir)
        
        # 保存最佳參數
        with open(os.path.join(save_dir, 'best_params.txt'), 'w') as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Training completed successfully")
        logger.info(f"Results saved to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()