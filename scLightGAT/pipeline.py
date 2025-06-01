# 修改 pipeline.py
# 路徑: scLightGAT/pipeline.py

import argparse
import os
import scanpy as sc
import torch
from scLightGAT.logger_config import setup_logger
from scLightGAT.training.model_manager import CellTypeAnnotator 
from scLightGAT.visualization.visualization import plot_umap_by_label

logger = setup_logger(__name__)

def train_pipeline(train_path: str, test_path: str, output_path: str, model_dir: str, train_dvae: bool, use_gat: bool, dvae_epochs: int, gat_epochs: int, hierarchical: bool = False):
    logger.info("[TRAIN MODE] Starting training pipeline")
    logger.info(f"Train data: {train_path}")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Output dir: {output_path}")
    logger.info(f"Model save dir: {model_dir}")
    logger.info(f"Train DVAE: {train_dvae}, Use GAT: {use_gat}, DVAE Epochs: {dvae_epochs}, GAT Epochs: {gat_epochs}")
    logger.info(f"Hierarchical mode: {hierarchical}")

    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)
    adata_train.raw = adata_train.copy()
    adata_test.raw = adata_test.copy()
    
    
    annotator = CellTypeAnnotator(
        use_dvae=train_dvae,
        use_hvg=True,
        hierarchical=hierarchical,
        dvae_params={'epochs': dvae_epochs},
        gat_epochs=gat_epochs
    )

    adata_result, dvae_losses, gat_losses = annotator.run_pipeline(
        adata_train=adata_train,
        adata_test=adata_test,
        save_visualizations=True
    )

    adata_result.write(os.path.join(output_path, "adata_with_predictions.h5ad"))
    logger.info("[TRAIN MODE] Training pipeline completed")

def predict_pipeline(train_path: str, test_path: str, output_path: str, model_dir: str):
    logger.info("[PREDICT MODE] Running inference pipeline")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Using models from: {model_dir}")

    adata_test = sc.read_h5ad(test_path)
    
    
    annotator = CellTypeAnnotator()

    adata_result = annotator.run_inference(
        adata_test=adata_test,
        model_dir=model_dir
    )

    adata_result.write(os.path.join(output_path, "adata_predicted.h5ad"))
    logger.info("[PREDICT MODE] Prediction pipeline completed")

def main():
    parser = argparse.ArgumentParser(description="scLightGAT: Cell Type Annotation Pipeline")
    parser.add_argument('--train', type=str, required=True, help='Path to training .h5ad file')
    parser.add_argument('--test', type=str, required=True, help='Path to testing .h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Directory to save results')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'visualize'], help='Execution mode')
    parser.add_argument('--model_dir', type=str, default='models/', help='Path to save/load models')
    parser.add_argument('--train_dvae', action='store_true', help='Flag to enable DVAE training (default: False)')
    parser.add_argument('--use_gat', action='store_true', help='Flag to enable GAT refinement after LightGBM')
    parser.add_argument('--dvae_epochs', type=int, default=15, help='Number of epochs for DVAE training')
    parser.add_argument('--gat_epochs', type=int, default=300, help='Number of epochs for GAT training')
    parser.add_argument('--hierarchical', action='store_true', help='Enable hierarchical classification')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == 'train':
        train_pipeline(args.train, args.test, args.output, args.model_dir, 
                       args.train_dvae, args.use_gat, args.dvae_epochs, args.gat_epochs,
                       args.hierarchical)
    elif args.mode == 'predict':
        predict_pipeline(args.train, args.test, args.output, args.model_dir)
    elif args.mode == 'visualize':
        visualize_pipeline(args.test, args.output)
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    main()