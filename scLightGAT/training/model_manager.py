import logging
import optuna
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Dict, Tuple, Any
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from scLightGAT.models.lightgbm_model import LightGBMModel
from scLightGAT.models.dvae_model import DVAE
from scLightGAT.models.gat_model import GATModel
from scLightGAT.preprocess.prediction_preprocess import prepare_data, balance_classes
from scLightGAT.visualization.visualization import plot_prediction_comparison, create_dvae_visualizations, plot_training_losses
from scLightGAT.preprocess.preprocess import visualization_process




from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Tuple, Any
from torch.cuda.amp import autocast, GradScaler
import torch
import scanpy as sc
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch.nn.functional as F
from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)



scaler = torch.cuda.amp.GradScaler()


class ModelManager:
    def __init__(self, input_dim: Optional[int] = None, 
                 use_default_lightgbm_params: bool = True, 
                 use_dvae: bool = True,
                 use_hvg: bool = True, 
                 dvae_params: Optional[Dict] = None,
                 gat_epochs: int = 300): 
        self.use_dvae = use_dvae
        self.use_hvg = use_hvg 
        self.lightgbm = LightGBMModel(use_default_params=use_default_lightgbm_params)
        self.dvae = None
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gat_epochs = gat_epochs 

        # default dvae
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
        self.dvae_params = self.default_dvae_params.copy()
        if dvae_params:
            self.dvae_params.update(dvae_params)

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

    def train_lightgbm(self, X: Any, y: Any, n_trials: int = 12, group_name: str = None, class_names: Any = None):
        """
        Train LightGBM model
        
        Args:
            X: Feature data
            y: Label data
            n_trials: Number of trials for optuna optimization
            group_name: Optional name of the cell group for subtype training
            class_names: Original class names for confusion matrix
        """
        logger.info("🎄Starting LightGBM training🎄")
        if not self.lightgbm.use_default_params:
            # using optuna
            self.lightgbm.optimize_params(X, y, n_trials=n_trials)
        self.lightgbm.train(X, y, group_name=group_name, class_names=class_names)
        logger.info("🎄LightGBM training completed🎄")

    def predict_lightgbm(self, X: Any) -> Any:
        """
        Make predictions using LightGBM model
        :param X: Data to predict
        :return: Predictions
        """
        return self.lightgbm.predict(X)

    def predict_proba_lightgbm(self, X: Any) -> Any:
        """
        Predict probabilities using LightGBM model
        :param X: Data to predict
        :return: Predicted probabilities
        """
        return self.lightgbm.predict_proba(X)
    def calculate_cluster_loss(self, z: torch.Tensor, 
                         labels: torch.Tensor, 
                         temperature: float) -> torch.Tensor:
        """Calculate clustering loss within batch"""
        z = F.normalize(z, dim=1)
        
        similarity_matrix = torch.matmul(z, z.T)
        
        labels = labels[:z.size(0)]
        labels = labels.view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float()
        mask = mask.fill_diagonal_(0)
        
        pos_pairs = mask * similarity_matrix
        pos_mask = (mask > 0)
        if pos_mask.sum() > 0:
            pos_mean = pos_pairs[pos_mask].mean()
        else:
            return torch.tensor(0., device=z.device)
        
        neg_mask = 1 - mask
        neg_pairs = neg_mask * similarity_matrix
        neg_pairs = neg_pairs[neg_mask.bool()].mean()
        
        loss = -torch.log(
            torch.exp(pos_mean / temperature) / 
            (torch.exp(pos_mean / temperature) + 
            torch.exp(neg_pairs / temperature) + 1e-6)
        )
        
        return loss
    def train_dvae(self, data_loader: Any, X_train_balanced: pd.DataFrame, 
               y_train_balanced: np.ndarray, epochs: int = 20, 
               learning_rate: float = 1e-3, kl_weight: float = 0.03) -> tuple:
        """
        Train DVAE with updated logic including cluster loss
        """
        logger.info("💫Starting DVAE training with cluster loss💫")
        
        self.dvae.to(self.device)
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
        
        best_loss = float('inf')
        best_model = None
        no_improve = 0
        patience = 15
        epoch_losses = []

        for epoch in range(epochs):
            self.dvae.train()
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                batch = batch[0].to(self.device)
                optimizer.zero_grad()
                
                x_hat, mean, log_var, z = self.dvae(batch)
                
                # Get corresponding batch labels
                start_idx = batch_idx * len(batch)
                end_idx = min((batch_idx + 1) * len(batch), len(y_train_balanced))
                batch_labels = torch.tensor(
                    y_train_balanced[start_idx:end_idx], 
                    device=self.device
                )
                
                reconstruction_loss = F.mse_loss(x_hat, batch)
                kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                
                # Add cluster loss after initial epochs
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
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dvae.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                del x_hat, mean, log_var, z, loss
                torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / len(data_loader)
            epoch_losses.append(avg_loss)
            
            
            
            scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = deepcopy(self.dvae.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                           f"Loss: {avg_loss:.4f}, "
                           f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
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
        
        original_latent_features = np.concatenate(original_latent_features, axis=0)
        latent_cols = [f'latent_{i}' for i in range(original_latent_features.shape[1])]
        latent_df = pd.DataFrame(
            original_latent_features, 
            index=X_train_balanced.index,
            columns=latent_cols
        )

        logger.info("💫DVAE training completed💫")
        return self.dvae, latent_df, epoch_losses


    def calculate_reconstruction_loss(self, X_tensor: torch.Tensor, batch_size: int = 1000) -> float:
        total_loss = 0
        n_batches = 0
        for i in range(0, X_tensor.shape[0], batch_size):
            batch = X_tensor[i:i+batch_size].to(self.device)
            x_hat, _, _, _ = self.dvae(batch)
            loss = nn.MSELoss()(x_hat, batch)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def encode_dvae(self, X: pd.DataFrame) -> np.ndarray:
        self.dvae.eval()
        with torch.no_grad():
            encoded = []
            for i in range(0, len(X), 1000):
                batch = X.iloc[i:i+1000].values
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                _, _, _, z = self.dvae(batch_tensor)
                encoded.append(z.cpu().numpy())
        return np.concatenate(encoded, axis=0)
    
    def process_dvae(self, data_loader: Any, X: Any, latent_dim) -> Tuple[np.ndarray, float]:
        self.dvae.eval()
        latent_features = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data[0].to(self.device)
                _, _, _, z = self.dvae(data)
                latent_features.append(z.cpu().numpy())
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            reconstruction_loss = self.calculate_reconstruction_loss(X_tensor)
        
        print(f"Reconstruction Loss (MSE) on Training Set (Latent Dimension: {latent_dim}): {reconstruction_loss:.4f}")
        latent_features = np.concatenate(latent_features, axis=0)
        return latent_features, reconstruction_loss
    

    def prepare_gat_data(self, adata, lightgbm_probs):
        logger.info("💾Preparing data for GAT")
        adata.layers["log_transformed"] = np.log1p(adata.X)
        matrix = adata.to_df(layer="log_transformed")
        adata_pd = pd.DataFrame(adata.obs)
        matrix["Cell_type"] = adata_pd['Celltype_training']
        prominent_gene_matrix = matrix.drop(["Cell_type"], axis=1)
        combine_features = prominent_gene_matrix.values.astype(float)
        
        umap_coords = adata.obsm['X_umap'].astype(float)
        leiden_labels = adata.obs['leiden'].astype(int).values.astype(float)
        
        leiden_labels_matrix = np.expand_dims(leiden_labels, axis=1)
        node_features = np.hstack([umap_coords, combine_features, leiden_labels_matrix, lightgbm_probs])

        connectivities = adata.obsp['connectivities'].tocoo()
        distances = adata.obsp['distances'].tocoo()
        distance_dict = {(row, col): dist_weight for row, col, dist_weight in zip(distances.row, distances.col, distances.data)}

        G = nx.Graph()
        for i in range(node_features.shape[0]):
            G.add_node(i, features=node_features[i])

        for row, col, conn_weight in zip(connectivities.row, connectivities.col, connectivities.data):
            dist_weight = distance_dict.get((row, col), None)
            if dist_weight is not None:
                G.add_edge(row, col, connect_weight=conn_weight, dist_weight=dist_weight)

        edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        return data


    def init_gat(self, in_channels: int, out_channels: int):
        """
        Initialize GAT model
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        self.gat = GATModel(in_channels, out_channels)
        logger.info("GAT model initialized")

    def train_gat(self, data: Any, lightgbm_preds: Any, num_epochs: int = 300, lr: float = 0.0005, weight_decay: float = 1e-5) -> list:
        """
        Train GAT model
        :param data: Graph data
        :param lightgbm_preds: LightGBM predictions
        :param num_epochs: Number of training epochs
        :param lr: Learning rate
        :param weight_decay: Weight decay
        :return: Training losses
        """
        if self.gat is None:
            raise ValueError("GAT model not initialized. Call init_gat first.")
        logger.info("Starting GAT training")
        lightgbm_preds = np.array(lightgbm_preds).astype(np.int64)
        losses = self.gat.train_model(data, lightgbm_preds, num_epochs, lr, weight_decay)
        logger.info("GAT training completed")
        return losses

    def evaluate_gat(self, data: Any, lightgbm_preds: Any, lightgbm_classes: Any) -> Tuple[Any, float]:
        """
        Evaluate GAT model
        :param data: Graph data
        :param lightgbm_preds: LightGBM predictions
        :param lightgbm_classes: LightGBM classes
        :return: Predictions and accuracy
        """
        if self.gat is None:
            raise ValueError("GAT model not initialized. Call init_gat first.")
        return self.gat.evaluate(data, lightgbm_preds, lightgbm_classes)

    def save_models(self, path: str):
        """
        Save all models
        :param path: Save path
        """
        torch.save({
            'dvae_state_dict': self.dvae.state_dict(),
            'gat_state_dict': self.gat.state_dict() if self.gat else None,
            'lightgbm_model': self.lightgbm.model
        }, path)
        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """
        Load all models
        :param path: Load path
        """
        checkpoint = torch.load(path)
        self.dvae.load_state_dict(checkpoint['dvae_state_dict'])
        if checkpoint['gat_state_dict']:
            if self.gat is None:
                raise ValueError("GAT model not initialized. Call init_gat before loading.")
            self.gat.load_state_dict(checkpoint['gat_state_dict'])
        self.lightgbm.model = checkpoint['lightgbm_model']
        logger.info(f"Models loaded from {path}")


    def plot_feature_importance(self, model, feature_names, save_path=None):
        """
        Plots feature importance for the LightGBM model.

        Args:
            model: Trained LightGBM model.
            feature_names: List of feature names corresponding to model input.
            save_path: If provided, saves the plot to the given path.
        """
        # Extract feature importance
        importance = model.booster_.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feature_importance_df.head(50),  # Show top 50 features
            x='Importance',
            y='Feature',
            palette='viridis'
        )
        plt.title('Top 50 Feature Importances (LightGBM)', fontsize=14)
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.ylabel('Feature Name', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    def transform_adata(self, adata, check_type):
        """
        Transform an AnnData object to include the specified annotations.

        Args:
            adata: AnnData object to transform.
            check_type: Annotation to use for the transformation.

        Returns:
            Transformed Pandas DataFrame.
        """
        logger.info(f"Transforming AnnData object for {check_type}")
        if "log_transformed" not in adata.layers:
            adata.layers["log_transformed"] = np.log1p(adata.X)
        matrix = adata.to_df(layer="log_transformed")
        adata_pd = pd.DataFrame(adata.obs)
        matrix["Cell_type"] = adata_pd[check_type]
        return matrix
    def run_lightgbm_pipeline(self, adata_train: Any, adata_test: Any, 
                               dvae_epochs: int,
                               use_dvae: Optional[bool] = None,
                               use_hvg: Optional[bool] = None) -> Dict[str, Any]:
        """
        LightGBM pipeline with configurable DVAE integration.

        Args:
            adata_train: AnnData object for training data.
            adata_test: AnnData object for testing data.
            dvae_epochs: Number of epochs for DVAE training.
            use_dvae: Whether to use DVAE (overrides class default).
            use_hvg: Whether to use HVG feature selection (overrides class default).
        
        Returns:
            Dictionary with results and predictions.
        """
        logger.info("🌟 Starting First-Stage Prediction 🌟")
        
        current_use_dvae = use_dvae if use_dvae is not None else self.use_dvae
        data = prepare_data(adata_train, adata_test)
        adata_test_dge = data['adata_test_dge']

        if current_use_dvae:
            # Train and prepare DVAE
            logger.info("Training DVAE...")
            X_hvg, y_hvg, input_dim, dataloader, encoder_dvae = data['dvae']
            
            if self.dvae is None:
                self.initialize_dvae(input_dim)
                
            _, train_latent_df, dvae_losses = self.train_dvae(
                dataloader,
                pd.DataFrame(X_hvg),
                y_hvg,
                epochs=self.dvae_params['epochs'],
                learning_rate=self.dvae_params['learning_rate'],
                kl_weight=self.dvae_params['kl_weight']
            )
            
            # Prepare DVAE test features
            test_matrix = data['adata_test_hvg'].to_df(layer="log_transformed")
            test_latent_features = self.encode_dvae(pd.DataFrame(test_matrix.values))
            latent_cols = [f'latent_{i}' for i in range(test_latent_features.shape[1])]
            test_latent_df = pd.DataFrame(
                test_latent_features, 
                index=test_matrix.index,
                columns=latent_cols
            )
            # Prepare combined features for training and testing
            dge_train_df = self.transform_adata(data['adata_train_dge'], "Celltype_training")
            dge_test_df = self.transform_adata(data['adata_test_dge'], "Celltype_training")

            dge_train_matrix = dge_train_df.drop("Cell_type", axis=1)
            dge_test_matrix = dge_test_df.drop("Cell_type", axis=1)

            train_latent_df = train_latent_df.loc[dge_train_matrix.index]
            test_latent_df = test_latent_df.loc[dge_test_matrix.index]

            X_train_combined = pd.concat([dge_train_matrix, train_latent_df], axis=1)
            X_test_combined = pd.concat([dge_test_matrix, test_latent_df], axis=1)

            X_train_balanced, y_balanced = balance_classes(
                X_train_combined,
                y_hvg,
                target_count=self.dvae_params['balanced_count']
            )
            encoder = encoder_dvae

        else:
            
            X_train_dge, y_balanced, encoder = data['lightgbm'] 
            adata_test_dge_df = self.transform_adata(data['adata_test_dge'], "Celltype_training")
            X_test_dge = adata_test_dge_df.drop("Cell_type", axis=1)

        
        self.train_lightgbm(X_train_balanced if current_use_dvae else X_train_dge, 
                            y_balanced, 
                            class_names=encoder.classes_)

        
        self.plot_feature_importance(
            self.lightgbm.model,
            feature_names=(X_train_combined if current_use_dvae else X_train_dge).columns,
            save_path="lightgbm_feature_importance.png"
        )

        
        lightgbm_preds_encoded = self.predict_lightgbm(X_test_combined if current_use_dvae else X_test_dge)
        lightgbm_probs = self.predict_proba_lightgbm(X_test_combined if current_use_dvae else X_test_dge)
        lightgbm_preds = encoder.inverse_transform(lightgbm_preds_encoded)

        
        adata_test_dge.obs['LightGBM_prediction'] = lightgbm_preds
        adata_test_dge.obsm['LightGBM_probs'] = lightgbm_probs

        
        logger.info("🌟 First-Stage Prediction Completed 🌟")
        
        return {
            'adata_test_dge': adata_test_dge,
            'lightgbm_preds': lightgbm_preds, 
            'lightgbm_preds_encoded': lightgbm_preds_encoded, 
            'lightgbm_probs': lightgbm_probs,
            'dvae_losses': dvae_losses if current_use_dvae else None,
            'latent_features': train_latent_df if current_use_dvae else None,
            'raw_cell_types': data['dvae'][1] if current_use_dvae else None,
            'lightgbm_encoder': encoder  
        }
    def run_gat_pipeline(self, adata_test_dge: Any, lightgbm_preds: Any, lightgbm_probs: Any, batch_key: Any = None, encoder: Optional[Any] = None) -> Tuple[Any, list[float]]:
        logger.info("\U0001F31FStarting Second-Stage prediction\U0001F31F")

        adata_test_dge = visualization_process(adata_test_dge, res=1.0, batch_key=batch_key)
        gat_data = self.prepare_gat_data(adata_test_dge, lightgbm_probs)

        # Label 編碼（使用現有整數 label）
        lightgbm_preds_idx = np.array(lightgbm_preds)

        # 取得對應的字串類別名稱
        if encoder is not None:
            unique_classes = encoder.inverse_transform(np.unique(lightgbm_preds_idx))
        else:
            raise ValueError("Encoder is required to decode label names for GAT evaluation.")

        self.init_gat(in_channels=gat_data.x.shape[1], out_channels=len(unique_classes))
        losses = self.train_gat(gat_data, lightgbm_preds_idx)
        pred_np, accuracy = self.evaluate_gat(gat_data, lightgbm_preds_idx, unique_classes)

        adata_test_dge.obs['GAT_pred'] = unique_classes[pred_np]
        logger.info(f"\U0001F31FSecond-Stage prediction completed\U0001F31F. GAT accuracy: {accuracy:.4f}")
        return adata_test_dge, losses

    def run_pipeline(self, adata_train: Any, adata_test: Any, 
                    dvae_epochs: Optional[int] = None,
                    gat_epochs: Optional[int] = None,
                    use_dvae: Optional[bool] = None,
                    use_hvg: Optional[bool] = None,
                    n_hvgs: Optional[int] = None,
                    save_visualizations: bool = True) -> Tuple[Any, List[float], List[float]]:
        """
        Run complete pipeline with visualizations
        
        Args:
            adata_train: Training data
            adata_test: Test data
            dvae_epochs: Override default DVAE training epochs
            gat_epochs: Override default GAT training epochs
            use_dvae: Whether to use DVAE
            use_hvg: Whether to use HVG
            n_hvgs: Number of HVGs
            save_visualizations: Save visualization images
        """
        current_use_dvae = use_dvae if use_dvae is not None else self.use_dvae

        if dvae_epochs is not None:
            self.dvae_params['epochs'] = dvae_epochs

        if n_hvgs is not None:
            self.dvae_params['n_hvgs'] = n_hvgs
        if gat_epochs is not None:
            self.gat_epochs = gat_epochs 

        # Run LightGBM pipeline
        lightgbm_results = self.run_lightgbm_pipeline(
            adata_train,
            adata_test,
            self.dvae_params['epochs'],
            use_dvae=current_use_dvae,
            use_hvg=use_hvg
        )

        # Run GAT pipeline
        gat_results, gat_losses = self.run_gat_pipeline(
            lightgbm_results['adata_test_dge'],
            lightgbm_results['lightgbm_preds_encoded'], 
            lightgbm_results['lightgbm_probs'],
            batch_key=None, 
            encoder=lightgbm_results['lightgbm_encoder'] 
        )

        # Create visualizations if requested
        if save_visualizations:
            prediction_columns = ['Celltype_training', 'LightGBM_prediction', 'GAT_pred']
            plot_prediction_comparison(
                gat_results,
                prediction_columns,
                figsize=(20, 8)
            )
            if current_use_dvae and lightgbm_results.get('dvae_losses') is not None:
                latent_df = lightgbm_results['latent_features']
                cell_types = adata_train.obs['Celltype_training'].loc[latent_df.index]
                create_dvae_visualizations(
                    latent_features=latent_df.values,
                    cell_types=cell_types,
                    title="DVAE" + (" with HVG" if use_hvg else " without HVG"),
                    dvae_losses=lightgbm_results['dvae_losses'],
                    save_path='dvae_analysis.png'
                )
            if gat_losses:
                plot_training_losses(
                    gat_losses,
                    model_name="GAT",
                    save_path="gat_training_loss.png"
                )

        return gat_results, lightgbm_results.get('dvae_losses', None), gat_losses
    def run_inference(self, adata_test: AnnData, model_dir: str) -> AnnData:
        self.load_models(model_dir)
        
        # Prepare X_dvae if DVAE exists
        if self.dvae:
            test_matrix = np.log1p(adata_test.X)
            test_latent_features = self.encode_dvae(pd.DataFrame(test_matrix))
            adata_test.obsm['X_dvae'] = test_latent_features

        # Extract prominent genes
        if 'prominent' not in adata_test.var.columns:
            raise ValueError("Missing 'prominent' gene annotation in adata.var.")
        
        prominent_genes = adata_test.var.index[adata_test.var['prominent'] == True].tolist()
        X_prominent = adata_test[:, prominent_genes].X
        X_dvae = adata_test.obsm.get('X_dvae', None)
        
        if isinstance(X_prominent, np.ndarray) and X_prominent.ndim == 1:
            X_prominent = X_prominent.reshape(-1, 1)
        if X_dvae is not None and X_dvae.ndim == 1:
            X_dvae = X_dvae.reshape(-1, 1)

        X_combined = np.hstack([X_dvae, X_prominent]) if X_dvae is not None else X_prominent

        y_pred = self.lightgbm.predict(X_combined)
        y_labels = self.lightgbm.encoder.inverse_transform(y_pred)
        adata_test.obs['scLightGAT_annotation'] = y_labels


        if self.gat:
            probs = self.lightgbm.predict_proba(X_combined)
            gat_data = self.prepare_gat_data(adata_test, probs)
            pred_np, _ = self.evaluate_gat(gat_data, y_pred, np.unique(y_pred))
            adata_test.obs['scLightGAT_annotation'] = pred_np

        return adata_test

    
    
    
class HierarchicalModelManager(ModelManager):
    def run_subtype_pipeline(self, adata_train, gat_results, balanced_counts=500, use_gat=False):
        """subtype_pipeline"""
        logger.info("🌟Starting subtype prediction pipeline🌟")
        
        # ensure string type
        # copy
        adata_test_sub = gat_results.copy()
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

        # The Board celltype need subtraining
        broad_types_to_process = ['CD4+T cells', 'CD8+T cells', 'B cells', 'Plasma cells', 'DC']
        
        all_predictions = []
        all_probabilities = []
        prediction_masks = []
        
        for broad_type in broad_types_to_process:
            logger.info(f"\nProcessing {broad_type}")
            
            group_train, group_test, test_mask, possible_subtypes = prepare_subtype_data_from_gat(
                adata_train, 
                adata_test_sub, 
                broad_type
            )
            
            if group_train is None or len(group_test) == 0:
                continue
            
            # train & predict
            X_train = self.transform_adata(group_train, 'Celltype_subtraining')
            X_test = self.transform_adata(group_test, 'Celltype_training')
            
            # ensure string type
            y_train = group_train.obs['Celltype_subtraining'].astype(str)
            
            X_balanced, y_balanced = balance_classes(
                X_train.drop('Cell_type', axis=1),
                y_train,
                target_count=balanced_counts
            )
            
            model = LightGBMModel(use_default_params=True)
            # Add group_name when training
            model.train(X_balanced, y_balanced, group_name=broad_type)  # Add this parameter
            
            predictions = model.predict(X_test.drop('Cell_type', axis=1))
            probabilities = model.predict_proba(X_test.drop('Cell_type', axis=1))
            
            # update lightgbm outcome
            mask_indices = adata_test_sub.obs.index[test_mask]
            adata_test_sub.obs.loc[mask_indices, 'subtype_pred_lightgbm'] = predictions
            
            # store outcome for gat
            all_predictions.extend(predictions)
            all_probabilities.append(probabilities)
            prediction_masks.append(test_mask)
            
        
            path_update = f"Broad: {broad_type} -> LightGBM: {predictions}"
            adata_test_sub.obs.loc[mask_indices, 'prediction_path'] = path_update
        if use_gat:
            # GAT refinement 
            combined_mask = np.any(prediction_masks, axis=0)
            if np.any(combined_mask):
                valid_cells = adata_test_sub[combined_mask].copy()
                valid_cells = visualization_process(valid_cells, res=1.0)
                
                # combine probabilities
                combined_probas = np.zeros((len(adata_test_sub), 0))
                for probas, mask in zip(all_probabilities, prediction_masks):
                    temp_probas = np.zeros((len(adata_test_sub), probas.shape[1]))
                    temp_probas[mask] = probas
                    combined_probas = np.hstack([combined_probas, temp_probas])
                
                # train GAT
                gat_data = self.prepare_gat_data(valid_cells, combined_probas[combined_mask])
                encoder = LabelEncoder()
                encoded_predictions = encoder.fit_transform(all_predictions)
                
                self.init_gat(
                    in_channels=gat_data.x.shape[1],
                    out_channels=len(np.unique(encoded_predictions))
                )
                self.train_gat(gat_data, encoded_predictions)
                
                # GAT predict
                final_preds, accuracy = self.evaluate_gat(
                    gat_data, 
                    encoded_predictions,
                    encoder.classes_
                )
                
                # update
                mask_indices = adata_test_sub.obs.index[combined_mask]
                final_predictions = encoder.inverse_transform(final_preds)
                adata_test_sub.obs.loc[mask_indices, 'subtype_pred_final'] = final_predictions
                
                # prediction path
                for idx, pred in zip(mask_indices, final_predictions):
                    current_path = adata_test_sub.obs.loc[idx, 'prediction_path']
                    adata_test_sub.obs.loc[idx, 'prediction_path'] = f"{current_path} -> GAT: {pred}"
            
        
            no_subtype_mask = ~combined_mask
            if np.any(no_subtype_mask):
                adata_test_sub.obs.loc[no_subtype_mask, 'prediction_path'] = (
                    "Broad (no subtype): " + 
                    adata_test_sub.obs.loc[no_subtype_mask, 'GAT_pred']
                )
        else:
            # directly use lightgbm prediction
            adata_test_sub.obs['final_pred'] = adata_test_sub.obs['subtype_pred_lightgbm']
            adata_test_sub.obs['prediction_path'] = (
                "Broad: " + adata_test_sub.obs['GAT_pred'].astype(str) + 
                " -> Final: " + adata_test_sub.obs['subtype_pred_lightgbm'].astype(str)
            )
        
            
        logger.info("Subtype prediction completed")
        
        return adata_test_sub