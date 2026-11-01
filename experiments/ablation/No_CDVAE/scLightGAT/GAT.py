"""Memory-bounded graph-attention refinement used by scLightGAT."""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from tqdm import tqdm

from .hparam import (
    GAT_CLASS_WEIGHT_EXPONENT,
    GAT_CLASS_WEIGHT_MAX,
    GAT_DROPOUT,
    GAT_EPOCHS,
    GAT_FEATURE_CHUNKS,
    GAT_GRADIENT_CLIP_NORM,
    GAT_HEADS,
    GAT_HIDDEN_CHANNELS,
    GAT_LEARNING_RATE,
    GAT_OUTPUT_HEADS,
    GAT_WEIGHT_DECAY,
)
from .random import seed_everything


class FeatureDecomposedGATConv(GATConv):
    """Split homogeneous GAT message propagation across feature channels."""

    def __init__(self, *args, feature_chunks=None, **kwargs):
        if feature_chunks is None:
            feature_chunks = GAT_FEATURE_CHUNKS
        super().__init__(*args, **kwargs)
        chunks = min(int(feature_chunks), int(self.out_channels))
        while self.out_channels % chunks != 0:
            chunks -= 1
        self.decomposed_layers = chunks

    def propagate(self, edge_index, size=None, **kwargs):
        features = kwargs.get("x")
        if isinstance(features, tuple):
            if features[0] is not features[1]:
                raise ValueError(
                    "FeatureDecomposedGATConv only supports homogeneous graphs"
                )
            kwargs["x"] = features[0]
        return MessagePassing.propagate(
            self, edge_index, size=size, **kwargs
        )


class GATModel(torch.nn.Module):
    """Two-layer graph-attention classifier with a residual connection."""

    def __init__(self, in_channels, out_channels, config=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.config = {
            "class_weight_exponent": GAT_CLASS_WEIGHT_EXPONENT,
            "class_weight_max": GAT_CLASS_WEIGHT_MAX,
            "dropout": GAT_DROPOUT,
            "epochs": GAT_EPOCHS,
            "heads": GAT_HEADS,
            "hidden_channels": GAT_HIDDEN_CHANNELS,
            "learning_rate": GAT_LEARNING_RATE,
            "weight_decay": GAT_WEIGHT_DECAY,
        }
        if config:
            self.config.update(deepcopy(config))

        hidden = int(self.config["hidden_channels"])
        heads = int(self.config["heads"])
        self.dropout = float(self.config["dropout"])
        self.gradient_clip_norm = float(GAT_GRADIENT_CLIP_NORM)
        self.conv1 = FeatureDecomposedGATConv(
            self.in_channels,
            hidden,
            heads=heads,
            dropout=self.dropout,
            feature_chunks=GAT_FEATURE_CHUNKS,
        )
        self.norm1 = torch.nn.LayerNorm(hidden * heads)
        self.conv2 = GATConv(
            hidden * heads,
            self.out_channels,
            heads=int(GAT_OUTPUT_HEADS),
            concat=False,
            dropout=self.dropout,
        )
        self.norm2 = torch.nn.LayerNorm(self.out_channels)
        self.residual = (
            torch.nn.Linear(self.in_channels, self.out_channels)
            if self.in_channels != self.out_channels
            else torch.nn.Identity()
        )

    def forward(self, data):
        features, edge_index = data.x, data.edge_index
        identity = features
        features = F.dropout(
            features, p=self.dropout, training=self.training
        )
        features = F.elu(self.norm1(self.conv1(features, edge_index)))
        features = F.dropout(
            features, p=self.dropout, training=self.training
        )
        features = (
            self.norm2(self.conv2(features, edge_index))
            + self.residual(identity)
        )
        return F.log_softmax(features, dim=1)

    def _class_weights(self, labels, train_mask):
        counts = torch.bincount(
            labels[train_mask], minlength=self.out_channels
        ).float()
        present = counts > 0
        weights = torch.zeros_like(counts)
        inverse_frequency = (
            counts[present].sum() / (present.sum() * counts[present])
        )
        weights[present] = inverse_frequency.pow(
            float(self.config["class_weight_exponent"])
        )
        weights[present] /= weights[present].mean()
        maximum = self.config.get("class_weight_max")
        if maximum is not None:
            weights[present] = weights[present].clamp(max=float(maximum))
        return weights

    def fit(self, data, labels, seed):
        seed_everything(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        data = data.to(device)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        train_mask = getattr(data, "train_mask", None)
        if train_mask is None:
            train_mask = torch.ones(
                data.num_nodes, dtype=torch.bool, device=device
            )

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        criterion = torch.nn.NLLLoss(
            weight=self._class_weights(labels, train_mask)
        )
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        losses = []

        self.train()
        iterator = tqdm(
            range(int(self.config["epochs"])),
            desc="Training GAT",
            leave=True,
            dynamic_ncols=True,
        )
        for _ in iterator:
            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=device.type, enabled=amp_enabled
            ):
                output = self(data)
                loss = criterion(output[train_mask], labels[train_mask])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            iterator.set_postfix(loss=f"{loss.item():.4f}")
        return losses

    def predict_proba(self, data):
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            output = self(data.to(device))
        return output.exp().cpu().numpy()

    def predict(self, data):
        return self.predict_proba(data).argmax(axis=1)
