import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        
        self.conv1 = GATConv(in_channels, 128, heads=8, dropout=0.5)  
        self.conv2 = GATConv(128 * 8, out_channels, heads=1, concat=False, dropout=0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.5, training=self.training)
        x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        
        return F.log_softmax(x, dim=1), attention_weights1[1], attention_weights2[1]

    def train_model(self, data, lightgbm_preds, num_epochs=100, lr=0.0005, weight_decay=1e-5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = data.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        lightgbm_preds = torch.tensor(lightgbm_preds, dtype=torch.long).to(device)

        losses = []
        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out, _, _ = self(data)
            loss = criterion(out, lightgbm_preds)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            torch.cuda.empty_cache() 
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        return losses

    def evaluate(self, data, lightgbm_preds, lightgbm_classes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            out, _, _ = self(data.to(device))
            _, pred = out.max(dim=1)
            
        pred_np = pred.cpu().numpy()
        lightgbm_preds_np = np.array(lightgbm_preds).astype(np.int64)

        accuracy = accuracy_score(lightgbm_preds_np, pred_np)
        report = classification_report(lightgbm_preds_np, pred_np, target_names=lightgbm_classes)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        return pred_np, accuracy
