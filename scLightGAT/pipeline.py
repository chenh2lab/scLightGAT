import argparse
import os
import scanpy as sc
import torch
from scLightGAT.logger_config import setup_logger
from scLightGAT.training.model_manager import ModelManager
from scLightGAT.visualization.visualization import plot_umap_by_label

logger = setup_logger(__name__)

def train_pipeline(train_path: str, test_path: str, output_path: str, model_dir: str, train_dvae: bool, use_gat: bool, dvae_epochs: int, gat_epochs: int):
    logger.info("[TRAIN MODE] Starting training pipeline")
    logger.info(f"Train data: {train_path}")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Output dir: {output_path}")
    logger.info(f"Model save dir: {model_dir}")
    logger.info(f"Train DVAE: {train_dvae}, Use GAT: {use_gat}, DVAE Epochs: {dvae_epochs}, GAT Epochs: {gat_epochs}")

    adata_train = sc.read_h5ad(train_path)
    adata_test = sc.read_h5ad(test_path)
    model_manager = ModelManager()

    # Set custom training parameters
    model_manager.dvae_params['epochs'] = dvae_epochs
    model_manager.gat_epochs = gat_epochs  # Custom attribute for GAT

    adata_result, dvae_losses, gat_losses = model_manager.run_pipeline(
        adata_train=adata_train,
        adata_test=adata_test,
        use_dvae=train_dvae,
        dvae_epochs=dvae_epochs,
        save_visualizations=True,
        gat_epochs=gat_epochs
    )

    adata_result.write(os.path.join(output_path, "adata_with_predictions.h5ad"))
    logger.info("[TRAIN MODE] Training pipeline completed")

def predict_pipeline(train_path: str, test_path: str, output_path: str, model_dir: str):
    logger.info("[PREDICT MODE] Running inference pipeline")
    logger.info(f"Train data: {train_path}")
    logger.info(f"Test data: {test_path}")
    logger.info(f"Using models from: {model_dir}")

    adata_test = sc.read_h5ad(test_path)
    model_manager = ModelManager()

    adata_result = model_manager.run_inference(
        adata_test=adata_test,
        model_dir=model_dir
    )

    adata_result.write(os.path.join(output_path, "adata_predicted.h5ad"))
    logger.info("[PREDICT MODE] Prediction pipeline completed")

def visualize_pipeline(input_path: str, output_path: str):
    logger.info("[VISUALIZE MODE] Generating UMAP and cluster plots")
    logger.info(f"Input: {input_path}")
    logger.info(f"Save to: {output_path}")

    adata = sc.read_h5ad(input_path)

    if 'X_umap' not in adata.obsm:
        logger.info("UMAP coordinates not found. Running dimensionality reduction...")
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    os.makedirs(output_path, exist_ok=True)

    if 'Celltype_training' in adata.obs:
        plot_umap_by_label(adata, label='Celltype_training', save_path=os.path.join(output_path, 'umap_Cell_type.png'))

    if 'scLightGAT_annotation' in adata.obs:
        plot_umap_by_label(adata, label='scLightGAT_annotation', save_path=os.path.join(output_path, 'umap_scLightGAT_annotation.png'))

    logger.info("UMAP plots saved.")

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

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == 'train':
        train_pipeline(args.train, args.test, args.output, args.model_dir, args.train_dvae, args.use_gat, args.dvae_epochs, args.gat_epochs)
    elif args.mode == 'predict':
        predict_pipeline(args.train, args.test, args.output, args.model_dir)
    elif args.mode == 'visualize':
        visualize_pipeline(args.test, args.output)
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":
    main()
