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

# class GATModel(torch.nn.Module):
#     """
#     Graph Attention Network model for refining cell type predictions.
#     Incorporates layer normalization, residual connections, and mixed precision training.
#     """
#     def __init__(self, in_channels, out_channels):
#         """
    #     Initialize GAT model with dual attention layers.
        
    #     Args:
    #         in_channels: Number of input feature dimensions
    #         out_channels: Number of output classes
    #     """
    #     super(GATModel, self).__init__()
        
    #     # First GAT layer with multi-head attention
    #     self.conv1 = GATConv(in_channels, 256, heads=16, dropout=0.5)
    #     self.norm1 = torch.nn.LayerNorm(128 * 8)
        
    #     # Second GAT layer with single-head attention
    #     self.conv2 = GATConv(128 * 8, out_channels, heads=1, concat=False, dropout=0.5)
    #     self.norm2 = torch.nn.LayerNorm(out_channels)
        
    #     # Residual connection to preserve input information
    #     self.residual = torch.nn.Linear(in_channels, out_channels) if in_channels != out_channels else torch.nn.Identity()
        
    # def forward(self, data):
    #     """
    #     Forward pass through GAT model.
        
    #     Args:
    #         data: PyTorch Geometric Data object containing node features and edge indices
            
    #     Returns:
    #         Tuple of (log_softmax outputs, attention weights for layer 1, attention weights for layer 2)
    #     """
    #     x, edge_index = data.x, data.edge_index
    #     identity = x  # Store original features for residual connection
        
    #     # First GAT layer
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
    #     x = self.norm1(x)  # Layer normalization for training stability
    #     x = F.elu(x)
        
    #     # Second GAT layer
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
    #     x = self.norm2(x)
        
    #     # Add residual connection
    #     x = x + self.residual(identity)
        
    #     return F.log_softmax(x, dim=1), attention_weights1[1], attention_weights2[1]
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        # Layer-1: multi-head attention; output dim = 256 * 16 (concat=True by default)
        self.conv1 = GATConv(in_channels, 256, heads=16, dropout=0.5)
        # Layer-2: single-head output; concat=False keeps output dim = out_channels
        self.conv2 = GATConv(256 * 16, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout before attention is common to regularize features
        x = F.dropout(x, p=0.5, training=self.training)
        # Return attention weights to visualize later
        x, attn_w1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x, attn_w2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        # IMPORTANT: return raw logits (no log_softmax); CrossEntropyLoss expects logits
        return x, attn_w1[1], attn_w2[1]
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

    def evaluate(self, data, lightgbm_preds, lightgbm_classes=None):
        """
        Lightweight evaluation for ablation: compute only accuracy against teacher labels.
        This intentionally skips classification_report / confusion_matrix to avoid
        label-set mismatches between train/test (e.g., missing classes in a batch).

        Args:
            data: PyTorch Geometric Data object
            lightgbm_preds: Encoded teacher labels from LightGBM (np.ndarray or list of ints)
            lightgbm_classes: (optional) full class name list; unused here

        Returns:
            Tuple of (predictions_encoded, accuracy_float)
        """
        import numpy as np
        from sklearn.metrics import accuracy_score
        import torch
        from torch.cuda.amp import autocast

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            with autocast():
                out, _, _ = self(data.to(device))
                _, pred = out.max(dim=1)

        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        y_true = np.asarray(lightgbm_preds, dtype=np.int64)

        # Accuracy only
        acc = accuracy_score(y_true, pred_np)
        logger.info(f"Accuracy (GAT vs LightGBM labels): {acc:.4f}")

        return pred_np, acc