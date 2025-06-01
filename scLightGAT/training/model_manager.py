import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Tuple, Any, List
from copy import deepcopy
from tqdm import tqdm
from anndata import AnnData
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from scLightGAT.models.lightgbm_model import LightGBMModel
from scLightGAT.models.dvae_model import DVAE
from scLightGAT.models.gat_model import GATModel
from scLightGAT.preprocess.prediction_preprocess import prepare_data, balance_classes, prepare_subtype_data_from_gat
from scLightGAT.visualization.visualization import plot_prediction_comparison, create_dvae_visualizations, plot_training_losses
from scLightGAT.preprocess.preprocess import visualization_process
from scLightGAT.logger_config import setup_logger

logger = setup_logger(__name__)

class CellTypeAnnotator:
    """
    Unified cell type annotation framework for scRNA-seq data using LightGBM-GAT.
    Integrates feature extraction, classification, and refinement in a single pipeline.
    """
    
    def __init__(self, 
                 use_dvae: bool = True,
                 use_hvg: bool = True, 
                 input_dim: Optional[int] = None,
                 use_default_lightgbm_params: bool = True,
                 dvae_params: Optional[Dict] = None,
                 gat_epochs: int = 300,
                 hierarchical: bool = False):
        """
        Initialize the CellTypeAnnotator.
        
        Args:
            use_dvae: Whether to use DVAE for feature extraction
            use_hvg: Whether to use highly variable genes
            input_dim: Input dimension for DVAE (can be set later)
            use_default_lightgbm_params: Whether to use default LightGBM parameters
            dvae_params: Custom parameters for DVAE
            gat_epochs: Number of epochs for GAT training
            hierarchical: Whether to enable hierarchical classification
        """
        # Core configuration
        self.use_dvae = use_dvae
        self.use_hvg = use_hvg
        self.hierarchical = hierarchical
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gat_epochs = gat_epochs
        
        # Initialize models
        self.lightgbm = LightGBMModel(use_default_params=use_default_lightgbm_params)
        self.dvae = None
        self.gat = None
        
        # Set default DVAE parameters
        self.default_dvae_params = {
            'latent_dim': 300,
            'cluster_weight': 1,
            'temperature': 0.01,
            'epochs': 15,
            'batch_size': 128,
            'balanced_count': 10000,
            'n_hvgs': 3000,
            'learning_rate': 1e-5,
            'kl_weight': 0.01,
            'noise_factor': 0.1
        }
        
        # Update with custom parameters if provided
        self.dvae_params = self.default_dvae_params.copy()
        if dvae_params:
            self.dvae_params.update(dvae_params)
        
        logger.info(f"CellTypeAnnotator initialized on device: {self.device}")
        logger.info(f"DVAE enabled: {self.use_dvae}, HVG enabled: {self.use_hvg}")
        logger.info(f"Hierarchical classification: {self.hierarchical}")
    
    def initialize_dvae(self, input_dim: int):
        """Initialize DVAE model with current parameters"""
        if self.dvae is None:
            self.input_dim = input_dim
            self.dvae = DVAE(
                input_dim=self.input_dim,
                latent_dim=self.dvae_params['latent_dim'],
                noise_factor=self.dvae_params['noise_factor'],
                cluster_weight=self.dvae_params['cluster_weight'],
                temperature=self.dvae_params['temperature']
            )
            logger.info(f"DVAE initialized with input_dim: {self.input_dim}")
        else:
            logger.warning("DVAE is already initialized.")
    
    def calculate_cluster_loss(self, z: torch.Tensor, labels: torch.Tensor, 
                               temperature: float) -> torch.Tensor:
        """
        Calculate clustering loss for contrastive learning.
        
        Args:
            z: Latent vectors
            labels: Cell type labels
            temperature: Temperature parameter for contrastive loss
            
        Returns:
            Contrastive loss value
        """
        # Normalize latent vectors
        z = F.normalize(z, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(z, z.T)
        
        # Ensure labels match batch size
        labels = labels[:z.size(0)]
        labels = labels.view(-1, 1)
        
        # Create mask for positive pairs (same class)
        mask = torch.eq(labels, labels.T).float()
        mask = mask.fill_diagonal_(0)  # Exclude self-comparisons
        
        # Calculate positive pair similarities
        pos_pairs = mask * similarity_matrix
        pos_mask = (mask > 0)
        
        # Handle edge case with no positive pairs
        if pos_mask.sum() > 0:
            pos_mean = pos_pairs[pos_mask].mean()
        else:
            return torch.tensor(0., device=z.device)
        
        # Calculate negative pair similarities (different classes)
        neg_mask = 1 - mask
        neg_pairs = neg_mask * similarity_matrix
        neg_pairs = neg_pairs[neg_mask.bool()].mean()
        
        # Calculate InfoNCE loss
        loss = -torch.log(
            torch.exp(pos_mean / temperature) / 
            (torch.exp(pos_mean / temperature) + 
             torch.exp(neg_pairs / temperature) + 1e-6)
        )
        
        return loss
    
    def train_dvae(self, data_loader: DataLoader, 
                  X_train_balanced: pd.DataFrame, 
                  y_train_balanced: np.ndarray, 
                  epochs: int = 20, 
                  learning_rate: float = 1e-5, 
                  kl_weight: float = 0.01) -> tuple:
        """
        Train DVAE with contrastive learning.
        
        Args:
            data_loader: DataLoader for training data
            X_train_balanced: Feature matrix for training
            y_train_balanced: Labels for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            kl_weight: Weight for KL divergence loss
            
        Returns:
            Tuple of (trained DVAE model, latent features DataFrame, training losses)
        """
        logger.info("Starting DVAE training with contrastive learning")
        
        # Set up mixed precision training
        scaler = GradScaler()
        
        # Move model to device
        self.dvae.to(self.device)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.dvae.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        # Training tracking
        best_loss = float('inf')
        best_model = None
        no_improve = 0
        patience = 15
        epoch_losses = []
        
        # Training loop
        for epoch in range(epochs):
            self.dvae.train()
            epoch_loss = 0
            
            # Process batches
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                batch = batch[0].to(self.device)
                optimizer.zero_grad()
                
                # Use mixed precision for forward pass
                with autocast():
                    x_hat, mean, log_var, z = self.dvae(batch)
                    
                    # Get batch labels
                    start_idx = batch_idx * len(batch)
                    end_idx = min((batch_idx + 1) * len(batch), len(y_train_balanced))
                    batch_labels = torch.tensor(
                        y_train_balanced[start_idx:end_idx], 
                        device=self.device
                    )
                    
                    # Calculate losses
                    reconstruction_loss = F.mse_loss(x_hat, batch)
                    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    
                    # Add contrastive loss after initial epochs
                    if epoch > 10:
                        cluster_loss = self.calculate_cluster_loss(
                            z,
                            batch_labels,
                            temperature=self.dvae_params['temperature'] 
                        )
                        loss = (reconstruction_loss + 
                               kl_weight * kl_divergence + 
                               self.dvae_params['cluster_weight'] * cluster_loss) 
                    else:
                        loss = reconstruction_loss + kl_weight * kl_divergence
                
                # Scaled backward pass for mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.dvae.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
                # Free memory
                del x_hat, mean, log_var, z, loss
                torch.cuda.empty_cache()
            
            # Track average loss
            avg_loss = epoch_loss / len(data_loader)
            epoch_losses.append(avg_loss)
            
            # Update scheduler
            scheduler.step()
            
            # Check for improvement
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = deepcopy(self.dvae.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Log progress
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                           f"Loss: {avg_loss:.4f}, "
                           f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Load best model
        if best_model is not None:
            self.dvae.load_state_dict(best_model)
        
        # Generate latent features
        self.dvae.eval()
        original_latent_features = []
        
        with torch.no_grad():
            X_tensor = torch.tensor(X_train_balanced.values, dtype=torch.float32)
            for i in range(0, len(X_tensor), 1000):
                batch = X_tensor[i:i + 1000].to(self.device)
                _, _, _, z = self.dvae(batch)
                original_latent_features.append(z.cpu().numpy())
        
        # Create DataFrame of latent features
        original_latent_features = np.concatenate(original_latent_features, axis=0)
        latent_cols = [f'latent_{i}' for i in range(original_latent_features.shape[1])]
        latent_df = pd.DataFrame(
            original_latent_features, 
            index=X_train_balanced.index,
            columns=latent_cols
        )
        
        logger.info("DVAE training completed")
        return self.dvae, latent_df, epoch_losses
    
    def encode_dvae(self, X: pd.DataFrame) -> np.ndarray:
        """
        Encode data using trained DVAE.
        
        Args:
            X: Input data to encode
            
        Returns:
            Latent representations
        """
        self.dvae.eval()
        with torch.no_grad():
            encoded = []
            for i in range(0, len(X), 1000):
                batch = X.iloc[i:i+1000].values
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                _, _, _, z = self.dvae(batch_tensor)
                encoded.append(z.cpu().numpy())
        return np.concatenate(encoded, axis=0)
    
    def transform_adata(self, adata: AnnData, check_type: str) -> pd.DataFrame:
        """
        Transform AnnData object to DataFrame with cell type annotations.
        
        Args:
            adata: AnnData object to transform
            check_type: Column name in adata.obs to use as cell type
            
        Returns:
            DataFrame with features and cell type
        """
        logger.info(f"Transforming AnnData for {check_type}")
        
        # Ensure log-transformed data is available
        if "log_transformed" not in adata.layers:
            adata.layers["log_transformed"] = np.log1p(adata.X)
            
        # Create DataFrame from log-transformed data
        matrix = adata.to_df(layer="log_transformed")
        adata_pd = pd.DataFrame(adata.obs)
        matrix["Cell_type"] = adata_pd[check_type]
        
        return matrix
    
    def prepare_gat_data(self, adata: AnnData, lightgbm_probs: np.ndarray) -> Data:
        """
        Prepare graph data for GAT from AnnData and LightGBM probabilities.
        
        Args:
            adata: Annotated data with UMAP coordinates
            lightgbm_probs: Prediction probabilities from LightGBM
            
        Returns:
            PyTorch Geometric Data object for GAT
        """
        logger.info("Preparing data for GAT")
        
        # Ensure log-transformed data is available
        if "log_transformed" not in adata.layers:
            adata.layers["log_transformed"] = np.log1p(adata.X)
            
        # Extract gene expression matrix
        matrix = adata.to_df(layer="log_transformed")
        adata_pd = pd.DataFrame(adata.obs)
        matrix["Cell_type"] = adata_pd['Celltype_training']
        prominent_gene_matrix = matrix.drop(["Cell_type"], axis=1)
        combine_features = prominent_gene_matrix.values.astype(float)
        
        # Get UMAP coordinates and Leiden labels
        umap_coords = adata.obsm['X_umap'].astype(float)
        leiden_labels = adata.obs['leiden'].astype(int).values.astype(float)
        
        # Combine features for node attributes
        leiden_labels_matrix = np.expand_dims(leiden_labels, axis=1)
        node_features = np.hstack([umap_coords, combine_features, leiden_labels_matrix, lightgbm_probs])
        
        # Extract graph structure from neighbor graph
        connectivities = adata.obsp['connectivities'].tocoo()
        distances = adata.obsp['distances'].tocoo()
        distance_dict = {(row, col): dist_weight for row, col, dist_weight in zip(distances.row, distances.col, distances.data)}
        
        # Create networkx graph
        G = nx.Graph()
        for i in range(node_features.shape[0]):
            G.add_node(i, features=node_features[i])
        
        # Add edges with weights
        for row, col, conn_weight in zip(connectivities.row, connectivities.col, connectivities.data):
            dist_weight = distance_dict.get((row, col), None)
            if dist_weight is not None:
                G.add_edge(row, col, connect_weight=conn_weight, dist_weight=dist_weight)
        
        # Convert to PyTorch Geometric Data
        edge_index = torch.tensor(np.array(list(G.edges)).T, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        return data
    
    def init_gat(self, in_channels: int, out_channels: int):
        """
        Initialize GAT model.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output classes
        """
        self.gat = GATModel(in_channels, out_channels)
        logger.info(f"GAT model initialized with {in_channels} in_channels and {out_channels} out_channels")
    
    def train_gat(self, data: Data, lightgbm_preds: np.ndarray, 
                 num_epochs: int = 300, lr: float = 0.0005, 
                 weight_decay: float = 1e-5) -> list:
        """
        Train GAT model.
        
        Args:
            data: PyTorch Geometric Data object
            lightgbm_preds: Predictions from LightGBM
            num_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            
        Returns:
            List of training losses
        """
        if self.gat is None:
            raise ValueError("GAT model not initialized. Call init_gat first.")
            
        logger.info("Starting GAT training")
        lightgbm_preds = np.array(lightgbm_preds).astype(np.int64)
        losses = self.gat.train_model(data, lightgbm_preds, num_epochs, lr, weight_decay)
        logger.info("GAT training completed")
        
        return losses
    
    def evaluate_gat(self, data: Data, lightgbm_preds: np.ndarray, 
                    lightgbm_classes: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate GAT model.
        
        Args:
            data: PyTorch Geometric Data object
            lightgbm_preds: Predictions from LightGBM
            lightgbm_classes: Class names
            
        Returns:
            Tuple of (predictions, accuracy)
        """
        if self.gat is None:
            raise ValueError("GAT model not initialized. Call init_gat first.")
            
        return self.gat.evaluate(data, lightgbm_preds, lightgbm_classes)
    
    def run_feature_extraction(self, adata_train: AnnData, adata_test: AnnData) -> Dict[str, Any]:
        """
        Run feature extraction pipeline.
        
        Args:
            adata_train: Training data
            adata_test: Testing data
            
        Returns:
            Dictionary with extracted features and metadata
        """
        logger.info("Starting feature extraction")
        
        # Prepare data
        data = prepare_data(adata_train, adata_test)
        adata_test_dge = data['adata_test_dge']
        
        if self.use_dvae:
            # Extract features with DVAE
            logger.info("Extracting features with DVAE")
            X_hvg, y_hvg, input_dim, dataloader, encoder_dvae = data['dvae']
            
            # Initialize DVAE if needed
            if self.dvae is None:
                self.initialize_dvae(input_dim)
                
            # Train DVAE
            _, train_latent_df, dvae_losses = self.train_dvae(
                dataloader,
                pd.DataFrame(X_hvg),
                y_hvg,
                epochs=self.dvae_params['epochs'],
                learning_rate=self.dvae_params['learning_rate'],
                kl_weight=self.dvae_params['kl_weight']
            )
            
            # Generate latent features for test data
            test_matrix = data['adata_test_hvg'].to_df(layer="log_transformed")
            test_latent_features = self.encode_dvae(pd.DataFrame(test_matrix.values))
            latent_cols = [f'latent_{i}' for i in range(test_latent_features.shape[1])]
            test_latent_df = pd.DataFrame(
                test_latent_features, 
                index=test_matrix.index,
                columns=latent_cols
            )
            
            # Combine DVAE features with DGE features
            dge_train_df = self.transform_adata(data['adata_train_dge'], "Celltype_training")
            dge_test_df = self.transform_adata(data['adata_test_dge'], "Celltype_training")
            
            dge_train_matrix = dge_train_df.drop("Cell_type", axis=1)
            dge_test_matrix = dge_test_df.drop("Cell_type", axis=1)
            
            # Align indices
            train_latent_df = train_latent_df.loc[dge_train_matrix.index]
            test_latent_df = test_latent_df.loc[dge_test_matrix.index]
            
            # Create combined feature matrices
            X_train_combined = pd.concat([dge_train_matrix, train_latent_df], axis=1)
            X_test_combined = pd.concat([dge_test_matrix, test_latent_df], axis=1)
            
            # Balance classes
            X_train_balanced, y_balanced = balance_classes(
                X_train_combined,
                y_hvg,
                target_count=self.dvae_params['balanced_count']
            )
            
            encoder = encoder_dvae
            X_train_features = X_train_balanced
            X_test_features = X_test_combined
            
        else:
            # Use only DGE features
            X_train_dge, y_balanced, encoder = data['lightgbm']
            adata_test_dge_df = self.transform_adata(data['adata_test_dge'], "Celltype_training")
            X_test_dge = adata_test_dge_df.drop("Cell_type", axis=1)
            
            X_train_features = X_train_dge
            X_test_features = X_test_dge
            dvae_losses = None
            train_latent_df = None
        
        logger.info("Feature extraction completed")
        
        return {
            'adata_test_dge': adata_test_dge,
            'X_train': X_train_features,
            'X_test': X_test_features,
            'y_train': y_balanced,
            'encoder': encoder,
            'dvae_losses': dvae_losses,
            'latent_features': train_latent_df if self.use_dvae else None,
            'raw_cell_types': data['dvae'][1] if self.use_dvae else None
        }
    
    def run_classification(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run classification pipeline.
        
        Args:
            feature_data: Dictionary with features from run_feature_extraction
            
        Returns:
            Dictionary with classification results
        """
        logger.info("Starting classification")
        
        # Extract data
        X_train = feature_data['X_train']
        y_train = feature_data['y_train']
        X_test = feature_data['X_test']
        encoder = feature_data['encoder']
        adata_test_dge = feature_data['adata_test_dge']
        
        # Train LightGBM
        self.train_lightgbm(X_train, y_train, class_names=encoder.classes_)
        
        # Generate predictions
        lightgbm_preds_encoded = self.predict_lightgbm(X_test)
        lightgbm_probs = self.predict_proba_lightgbm(X_test)
        lightgbm_preds = encoder.inverse_transform(lightgbm_preds_encoded)
        
        # Store predictions in AnnData
        adata_test_dge.obs['LightGBM_prediction'] = lightgbm_preds
        adata_test_dge.obsm['LightGBM_probs'] = lightgbm_probs
        
        logger.info("Classification completed")
        
        return {
            'adata_test_dge': adata_test_dge,
            'lightgbm_preds': lightgbm_preds,
            'lightgbm_preds_encoded': lightgbm_preds_encoded,
            'lightgbm_probs': lightgbm_probs,
            'encoder': encoder
        }
    
    def run_refinement(self, classification_data: Dict[str, Any], 
                      batch_key: Optional[str] = None) -> Tuple[AnnData, List[float]]:
        """
        Run GAT refinement pipeline.
        
        Args:
            classification_data: Dictionary with results from run_classification
            batch_key: Optional batch key for UMAP generation
            
        Returns:
            Tuple of (refined AnnData, GAT losses)
        """
        logger.info("Starting refinement with GAT")
        
        # Extract data
        adata_test_dge = classification_data['adata_test_dge']
        lightgbm_preds = classification_data['lightgbm_preds_encoded']
        lightgbm_probs = classification_data['lightgbm_probs']
        encoder = classification_data['encoder']
        
        # Prepare visualization
        adata_test_dge = visualization_process(adata_test_dge, res=1.0, batch_key=batch_key)
        
        # Prepare GAT data
        gat_data = self.prepare_gat_data(adata_test_dge, lightgbm_probs)
        
        # Get unique classes
        if encoder is not None:
            unique_classes = encoder.inverse_transform(np.unique(lightgbm_preds))
        else:
            raise ValueError("Encoder is required for GAT refinement")
        
        # Initialize and train GAT
        self.init_gat(in_channels=gat_data.x.shape[1], out_channels=len(unique_classes))
        losses = self.train_gat(gat_data, lightgbm_preds, num_epochs=self.gat_epochs)
        
        # Evaluate and get refined predictions
        pred_np, accuracy = self.evaluate_gat(gat_data, lightgbm_preds, unique_classes)
        
        # Store refined predictions
        adata_test_dge.obs['GAT_pred'] = unique_classes[pred_np]
        
        logger.info(f"Refinement completed with accuracy: {accuracy:.4f}")
        
        return adata_test_dge, losses
    
    def run_subtype_prediction(self, adata_train: AnnData, 
                              adata_refined: AnnData, 
                              balanced_counts: int = 500, 
                              use_gat: bool = False) -> AnnData:
        """
        Run subtype prediction for specific cell types.
        
        Args:
            adata_train: Training data with subtype annotations
            adata_refined: Refined data from GAT with broad cell types
            balanced_counts: Target count for balanced sampling
            use_gat: Whether to use GAT for subtype refinement
            
        Returns:
            AnnData with subtype predictions
        """
        if not self.hierarchical:
            logger.warning("Hierarchical mode is disabled. Skipping subtype prediction.")
            return adata_refined
            
        logger.info("Starting subtype prediction")
        
        # Initialize results with broad cell types
        adata_test_sub = adata_refined.copy()
        adata_test_sub.obs['GAT_pred'] = adata_test_sub.obs['GAT_pred'].astype(str)
        adata_test_sub.obs['subtype_pred_lightgbm'] = pd.Series(
            adata_test_sub.obs['GAT_pred'].values,
            index=adata_test_sub.obs.index,
            dtype=str
        )
        adata_test_sub.obs['subtype_pred_final'] = pd.Series(
            adata_test_sub.obs['GAT_pred'].values,
            index=adata_test_sub.obs.index,
            dtype=str
        )
        adata_test_sub.obs['prediction_path'] = pd.Series(
            "",
            index=adata_test_sub.obs.index,
            dtype=str
        )
        
        # Define cell types to process for subtypes
        broad_types_to_process = ['CD4+T cells', 'CD8+T cells', 'B cells', 'Plasma cells', 'DC']
        
        # Storage for GAT refinement
        all_predictions = []
        all_probabilities = []
        prediction_masks = []
        
        # Process each broad cell type
        for broad_type in broad_types_to_process:
            logger.info(f"Processing {broad_type}")
            
            # Prepare subtype data
            group_train, group_test, test_mask, possible_subtypes = prepare_subtype_data_from_gat(
                adata_train, 
                adata_test_sub, 
                broad_type
            )
            
            # Skip if no data
            if group_train is None or len(group_test) == 0:
                continue
            
            # Transform data
            X_train = self.transform_adata(group_train, 'Celltype_subtraining')
            X_test = self.transform_adata(group_test, 'Celltype_training')
            
            # Get labels and balance classes
            y_train = group_train.obs['Celltype_subtraining'].astype(str)
            X_balanced, y_balanced = balance_classes(
                X_train.drop('Cell_type', axis=1),
                y_train,
                target_count=balanced_counts
            )
            
            # Train and predict
            model = LightGBMModel(use_default_params=True)
            model.train(X_balanced, y_balanced, group_name=broad_type)
            
            predictions = model.predict(X_test.drop('Cell_type', axis=1))
            probabilities = model.predict_proba(X_test.drop('Cell_type', axis=1))
            # Store predictions for current broad cell type
            mask_indices = adata_test_sub.obs.index[test_mask]
            adata_test_sub.obs.loc[mask_indices, 'subtype_pred_lightgbm'] = predictions
            
            # Record for potential GAT refinement
            all_predictions.extend(predictions)
            all_probabilities.append(probabilities)
            prediction_masks.append(test_mask)
            
            # Record prediction path
            path_update = f"Broad: {broad_type} -> LightGBM: {predictions}"
            adata_test_sub.obs.loc[mask_indices, 'prediction_path'] = path_update
        
        # Apply GAT refinement if requested
        if use_gat and all_predictions:
            logger.info("Applying GAT refinement to subtypes")
            
            # Create mask for all cells with subtype predictions
            combined_mask = np.any(prediction_masks, axis=0)
            
            if np.any(combined_mask):
                # Prepare data for refinement
                valid_cells = adata_test_sub[combined_mask].copy()
                valid_cells = visualization_process(valid_cells, res=1.0)
                
                # Combine probabilities from all broad types
                combined_probas = np.zeros((len(adata_test_sub), 0))
                for probas, mask in zip(all_probabilities, prediction_masks):
                    temp_probas = np.zeros((len(adata_test_sub), probas.shape[1]))
                    temp_probas[mask] = probas
                    combined_probas = np.hstack([combined_probas, temp_probas])
                
                # Train GAT on subtypes
                gat_data = self.prepare_gat_data(valid_cells, combined_probas[combined_mask])
                encoder = LabelEncoder()
                encoded_predictions = encoder.fit_transform(all_predictions)
                
                # Initialize and train GAT
                self.init_gat(
                    in_channels=gat_data.x.shape[1],
                    out_channels=len(np.unique(encoded_predictions))
                )
                self.train_gat(gat_data, encoded_predictions)
                
                # Generate refined predictions
                final_preds, accuracy = self.evaluate_gat(
                    gat_data, 
                    encoded_predictions,
                    encoder.classes_
                )
                
                # Update predictions
                mask_indices = adata_test_sub.obs.index[combined_mask]
                final_predictions = encoder.inverse_transform(final_preds)
                adata_test_sub.obs.loc[mask_indices, 'subtype_pred_final'] = final_predictions
                
                # Update prediction paths
                for idx, pred in zip(mask_indices, final_predictions):
                    current_path = adata_test_sub.obs.loc[idx, 'prediction_path']
                    adata_test_sub.obs.loc[idx, 'prediction_path'] = f"{current_path} -> GAT: {pred}"
                
                logger.info(f"Subtype GAT refinement completed with accuracy: {accuracy:.4f}")
            
            # Handle cells without subtypes
            no_subtype_mask = ~combined_mask
            if np.any(no_subtype_mask):
                adata_test_sub.obs.loc[no_subtype_mask, 'prediction_path'] = (
                    "Broad (no subtype): " + 
                    adata_test_sub.obs.loc[no_subtype_mask, 'GAT_pred']
                )
        else:
            # Use LightGBM predictions directly
            adata_test_sub.obs['subtype_pred_final'] = adata_test_sub.obs['subtype_pred_lightgbm']
            
            # Update prediction paths
            update_mask = adata_test_sub.obs['subtype_pred_lightgbm'] != adata_test_sub.obs['GAT_pred']
            if update_mask.any():
                adata_test_sub.obs.loc[update_mask, 'prediction_path'] = (
                    "Broad: " + adata_test_sub.obs.loc[update_mask, 'GAT_pred'].astype(str) + 
                    " -> Final: " + adata_test_sub.obs.loc[update_mask, 'subtype_pred_lightgbm'].astype(str)
                )
        
        logger.info("Subtype prediction completed")
        return adata_test_sub
    
    def predict_lightgbm(self, X: Any) -> Any:
        """
        Make predictions using LightGBM model.
        
        Args:
            X: Data to predict
            
        Returns:
            Predictions
        """
        return self.lightgbm.predict(X)
    
    def predict_proba_lightgbm(self, X: Any) -> Any:
        """
        Predict probabilities using LightGBM model.
        
        Args:
            X: Data to predict
            
        Returns:
            Predicted probabilities
        """
        return self.lightgbm.predict_proba(X)
    
    def train_lightgbm(self, X: Any, y: Any, n_trials: int = 12, 
                      group_name: str = None, class_names: Any = None):
        """
        Train LightGBM model.
        
        Args:
            X: Feature data
            y: Label data
            n_trials: Number of optimization trials
            group_name: Name of the cell group (for subtypes)
            class_names: Class names for confusion matrix
        """
        logger.info("Starting LightGBM training")
        
        if not self.lightgbm.use_default_params:
            # Optimize parameters
            self.lightgbm.optimize_params(X, y, n_trials=n_trials)
            
        # Train model
        self.lightgbm.train(X, y, group_name=group_name, class_names=class_names)
        
        logger.info("LightGBM training completed")
    
    def save_models(self, path: str):
        """
        Save all trained models.
        
        Args:
            path: Path to save models
        """
        torch.save({
            'dvae_state_dict': self.dvae.state_dict() if self.dvae else None,
            'gat_state_dict': self.gat.state_dict() if self.gat else None,
            'lightgbm_model': self.lightgbm.model if self.lightgbm else None
        }, path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """
        Load saved models.
        
        Args:
            path: Path to load models
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load DVAE if available
        if checkpoint['dvae_state_dict']:
            if self.dvae is None:
                raise ValueError("DVAE model not initialized. Initialize DVAE before loading.")
            self.dvae.load_state_dict(checkpoint['dvae_state_dict'])
            
        # Load GAT if available
        if checkpoint['gat_state_dict']:
            if self.gat is None:
                raise ValueError("GAT model not initialized. Initialize GAT before loading.")
            self.gat.load_state_dict(checkpoint['gat_state_dict'])
            
        # Load LightGBM if available
        if checkpoint['lightgbm_model']:
            self.lightgbm.model = checkpoint['lightgbm_model']
            
        logger.info(f"Models loaded from {path}")
    
    def run_pipeline(self, adata_train: AnnData, adata_test: AnnData, 
                    dvae_epochs: Optional[int] = None,
                    gat_epochs: Optional[int] = None,
                    use_dvae: Optional[bool] = None,
                    use_hvg: Optional[bool] = None,
                    n_hvgs: Optional[int] = None,
                    save_visualizations: bool = True,
                    run_subtypes: Optional[bool] = None) -> Tuple[AnnData, Optional[List[float]], Optional[List[float]]]:
        """
        Run complete annotation pipeline.
        
        Args:
            adata_train: Training data
            adata_test: Test data
            dvae_epochs: Override DVAE epochs
            gat_epochs: Override GAT epochs
            use_dvae: Whether to use DVAE
            use_hvg: Whether to use HVG
            n_hvgs: Number of HVGs to use
            save_visualizations: Whether to save visualizations
            run_subtypes: Whether to run subtype prediction
            
        Returns:
            Tuple of (annotated data, DVAE losses, GAT losses)
        """
        # Set parameters
        current_use_dvae = use_dvae if use_dvae is not None else self.use_dvae
        run_subtypes = run_subtypes if run_subtypes is not None else self.hierarchical
        
        # Update model parameters if needed
        if dvae_epochs is not None:
            self.dvae_params['epochs'] = dvae_epochs
        if n_hvgs is not None:
            self.dvae_params['n_hvgs'] = n_hvgs
        if gat_epochs is not None:
            self.gat_epochs = gat_epochs
        
        logger.info("Starting complete annotation pipeline")
        
        # Feature extraction
        feature_data = self.run_feature_extraction(
            adata_train,
            adata_test
        )
        
        # Classification
        classification_data = self.run_classification(feature_data)
        
        # Refinement
        refined_data, gat_losses = self.run_refinement(
            classification_data,
            batch_key=None
        )
        
        # Subtype prediction (optional)
        if run_subtypes:
            final_data = self.run_subtype_prediction(
                adata_train,
                refined_data,
                balanced_counts=500,
                use_gat=True
            )
        else:
            final_data = refined_data
        
        # Create visualizations
        if save_visualizations:
            self._create_visualizations(
                final_data, 
                feature_data.get('dvae_losses'), 
                gat_losses,
                current_use_dvae,
                use_hvg if use_hvg is not None else self.use_hvg,
                adata_train
            )
        
        logger.info("Annotation pipeline completed")
        return final_data, feature_data.get('dvae_losses'), gat_losses
    
    def _create_visualizations(self, annotated_data, dvae_losses, gat_losses, 
                           use_dvae, use_hvg, adata_train):
        """
        Create comprehensive visualizations for results.
        
        Args:
            annotated_data: Annotated AnnData object
            dvae_losses: DVAE training losses
            gat_losses: GAT training losses
            use_dvae: Whether DVAE was used
            use_hvg: Whether HVG was used
            adata_train: Training data for reference
        """
        import matplotlib.pyplot as plt
        import scanpy as sc
        
        # Ensure UMAP coordinates exist
        if 'X_umap' not in annotated_data.obsm:
            logger.info("Computing UMAP coordinates for visualization")
            sc.pp.neighbors(annotated_data)
            sc.tl.umap(annotated_data)
        # Create main prediction visualization
        if 'GAT_pred' in annotated_data.obs:
            logger.info("Creating UMAP visualization for cell type predictions")
            
            # Method 1: Using scanpy plotting with save parameter
            sc.settings.figdir = '.'  # Set output directory
            sc.pl.umap(
                annotated_data,
                color='GAT_pred',
                title="scLightGAT Cell Type Annotation",
                frameon=False,
                legend_loc='right margin',
                legend_fontsize=10,
                size=15,
                alpha=0.8,
                show=False,
                save='_scLightGAT.png'  # This will save with 'umap_GAT_pred_scLightGAT.png' filename
            )
        # Method 2: Using matplotlib directly for more control
            plt.figure(figsize=(10, 8))
            ax = sc.pl.umap(
                annotated_data,
                color='GAT_pred',
                title="scLightGAT Cell Type Annotation",
                frameon=False,
                legend_loc='right margin',
                legend_fontsize=10,
                size=15,
                alpha=0.8,
                show=False,
                return_fig=True
            ).axes[0]
            
            plt.tight_layout()
            plt.savefig('scLightGAT_direct.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # DVAE visualizations
        if use_dvae and dvae_losses is not None and 'latent_features' in annotated_data.uns:
            latent_df = annotated_data.uns['latent_features']
            if latent_df is not None and 'Celltype_training' in adata_train.obs:
                cell_types = adata_train.obs['Celltype_training'].loc[latent_df.index]
                create_dvae_visualizations(
                    latent_features=latent_df.values,
                    cell_types=cell_types,
                    title="DVAE" + (" with HVG" if use_hvg else " without HVG"),
                    dvae_losses=dvae_losses,
                    save_path='dvae_analysis.png'
                )
        
        # GAT training loss
        if gat_losses:
            plot_training_losses(
                gat_losses,
                model_name="GAT",
                save_path="gat_training_loss.png"
            )