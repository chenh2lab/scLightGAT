# gat_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)

class GATModel(torch.nn.Module):
    """
    Graph Attention Network model for refining cell type predictions.
    Incorporates layer normalization, residual connections, and mixed precision training.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT model with dual attention layers.
        
        Args:
            in_channels: Number of input feature dimensions
            out_channels: Number of output classes
        """
        super(GATModel, self).__init__()
        
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(in_channels, 128, heads=8, dropout=0.5)
        self.norm1 = torch.nn.LayerNorm(128 * 8)
        
        # Second GAT layer with single-head attention
        self.conv2 = GATConv(128 * 8, out_channels, heads=1, concat=False, dropout=0.5)
        self.norm2 = torch.nn.LayerNorm(out_channels)
        
        # Residual connection to preserve input information
        self.residual = torch.nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()
        
    def forward(self, data):
        """
        Forward pass through GAT model.
        
        Args:
            data: PyTorch Geometric Data object containing node features and edge indices
            
        Returns:
            Tuple of (log_softmax outputs, attention weights for layer 1, attention weights for layer 2)
        """
        x, edge_index = data.x, data.edge_index
        identity = x  # Store original features for residual connection
        
        # First GAT layer
        x = F.dropout(x, p=0.5, training=self.training)
        x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.norm1(x)  # Layer normalization for training stability
        x = F.elu(x)
        
        # Second GAT layer
        x = F.dropout(x, p=0.5, training=self.training)
        x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.norm2(x)
        
        # Add residual connection
        x = x + self.residual(identity)
        
        return F.log_softmax(x, dim=1), attention_weights1[1], attention_weights2[1]

    def train_model(self, data, lightgbm_preds, num_epochs=100, lr=0.0005, weight_decay=1e-5):
        """
        Train the GAT model using mixed precision and progress tracking.
        
        Args:
            data: PyTorch Geometric Data object
            lightgbm_preds: Initial predictions from LightGBM to refine
            num_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            
        Returns:
            List of training losses
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = data.to(device)

        # Optimizer setup
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Mixed precision training setup
        scaler = GradScaler()

        # Convert labels to tensor
        lightgbm_preds = torch.tensor(lightgbm_preds, dtype=torch.long).to(device)

        losses = []
        self.train()
        
        # Use tqdm for progress display
        pbar = tqdm(range(num_epochs), desc="Training GAT")
        
        for epoch in pbar:
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                out, _, _ = self(data)
                loss = criterion(out, lightgbm_preds)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Memory cleanup
            torch.cuda.empty_cache() 
            
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        return losses

    def evaluate(self, data, lightgbm_preds, lightgbm_classes):
        """
        Evaluate GAT model performance.
        
        Args:
            data: PyTorch Geometric Data object
            lightgbm_preds: Initial predictions from LightGBM to compare against
            lightgbm_classes: Class names for the classification report
            
        Returns:
            Tuple of (predictions, accuracy)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            # Use mixed precision for inference
            with autocast():
                out, _, _ = self(data.to(device))
                _, pred = out.max(dim=1)
            
        pred_np = pred.cpu().numpy()
        lightgbm_preds_np = np.array(lightgbm_preds).astype(np.int64)

        accuracy = accuracy_score(lightgbm_preds_np, pred_np)
        report = classification_report(lightgbm_preds_np, pred_np, target_names=lightgbm_classes)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(report)

        return pred_np, accuracy