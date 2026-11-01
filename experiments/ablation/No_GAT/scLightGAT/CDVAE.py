"""Contrastive denoising variational autoencoder used by scLightGAT."""

import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .hparam import (
    CDVAE_BATCH_NORM_MOMENTUM,
    CDVAE_BATCH_SIZE,
    CDVAE_CLUSTER_WARMUP_EPOCHS,
    CDVAE_CLUSTER_WEIGHT,
    CDVAE_DROPOUT,
    CDVAE_EARLY_STOPPING_PATIENCE,
    CDVAE_EPOCHS,
    CDVAE_GRADIENT_CLIP_NORM,
    CDVAE_HIDDEN_DIMS,
    CDVAE_INFERENCE_BATCH_SIZE,
    CDVAE_KL_WEIGHT,
    CDVAE_LATENT_DIM,
    CDVAE_LEARNING_RATE,
    CDVAE_NOISE_FACTOR,
    CDVAE_N_HVGS,
    CDVAE_OPTIMIZER_BETAS,
    CDVAE_OPTIMIZER_WEIGHT_DECAY,
    CDVAE_SCHEDULER_ETA_MIN,
    CDVAE_TEMPERATURE,
)
from .random import seed_everything


logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, architecture=None):
        super().__init__()
        config = {
            "hidden_dims": CDVAE_HIDDEN_DIMS,
            "dropout": CDVAE_DROPOUT,
            "batch_norm_momentum": CDVAE_BATCH_NORM_MOMENTUM,
        }
        if architecture:
            config.update(architecture)
        hidden_1, hidden_2, hidden_3 = config["hidden_dims"]
        momentum = float(config["batch_norm_momentum"])

        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc_mean = nn.Linear(hidden_3, latent_dim)
        self.fc_var = nn.Linear(hidden_3, latent_dim)
        self.activation = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(hidden_1, momentum=momentum)
        self.norm2 = nn.BatchNorm1d(hidden_2, momentum=momentum)
        self.norm3 = nn.BatchNorm1d(hidden_3, momentum=momentum)
        self.dropout = nn.Dropout(p=float(config["dropout"]))

    def forward(self, x):
        x = self.dropout(self.activation(self.norm1(self.fc1(x))))
        x = self.dropout(self.activation(self.norm2(self.fc2(x))))
        x = self.dropout(self.activation(self.norm3(self.fc3(x))))
        return self.fc_mean(x), self.fc_var(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, architecture=None):
        super().__init__()
        config = {
            "hidden_dims": CDVAE_HIDDEN_DIMS,
            "dropout": CDVAE_DROPOUT,
            "batch_norm_momentum": CDVAE_BATCH_NORM_MOMENTUM,
        }
        if architecture:
            config.update(architecture)
        hidden_1, hidden_2, hidden_3 = reversed(config["hidden_dims"])
        momentum = float(config["batch_norm_momentum"])

        self.fc1 = nn.Linear(latent_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, output_dim)
        self.activation = nn.LeakyReLU()
        self.norm1 = nn.BatchNorm1d(hidden_1, momentum=momentum)
        self.norm2 = nn.BatchNorm1d(hidden_2, momentum=momentum)
        self.norm3 = nn.BatchNorm1d(hidden_3, momentum=momentum)
        self.dropout = nn.Dropout(p=float(config["dropout"]))

    def forward(self, z):
        z = self.dropout(self.activation(self.norm1(self.fc1(z))))
        z = self.dropout(self.activation(self.norm2(self.fc2(z))))
        z = self.dropout(self.activation(self.norm3(self.fc3(z))))
        return self.fc4(z)


class CDVAE(nn.Module):
    """CDVAE model plus the training and encoding operations it owns."""

    def __init__(self, input_dim, config=None, architecture=None):
        super().__init__()
        self.config = {
            "batch_size": CDVAE_BATCH_SIZE,
            "cluster_weight": CDVAE_CLUSTER_WEIGHT,
            "epochs": CDVAE_EPOCHS,
            "kl_weight": CDVAE_KL_WEIGHT,
            "latent_dim": CDVAE_LATENT_DIM,
            "learning_rate": CDVAE_LEARNING_RATE,
            "n_hvgs": CDVAE_N_HVGS,
            "noise_factor": CDVAE_NOISE_FACTOR,
            "temperature": CDVAE_TEMPERATURE,
        }
        if config:
            self.config.update(deepcopy(config))
        latent_dim = int(self.config["latent_dim"])
        self.encoder = Encoder(input_dim, latent_dim, architecture)
        self.decoder = Decoder(latent_dim, input_dim, architecture)
        self.noise_factor = float(self.config["noise_factor"])

    def forward(self, x):
        encoder_input = (
            x + torch.randn_like(x) * self.noise_factor
            if self.training
            else x
        )
        mean, log_var = self.encoder(encoder_input)
        if self.training:
            standard_deviation = torch.exp(0.5 * log_var)
            latent = mean + torch.randn_like(standard_deviation) * standard_deviation
        else:
            latent = mean
        return self.decoder(latent), mean, log_var, latent

    @staticmethod
    def _contrastive_loss(latent, labels, temperature):
        with torch.autocast(device_type=latent.device.type, enabled=False):
            latent = F.normalize(latent.float(), dim=1)
            labels = labels.view(-1, 1)
            logits = torch.matmul(latent, latent.T) / float(temperature)
            non_self = ~torch.eye(
                latent.size(0), dtype=torch.bool, device=latent.device
            )
            positive = torch.eq(labels, labels.T) & non_self
            positive_counts = positive.sum(dim=1)
            valid = positive_counts > 0
            if not valid.any():
                return latent.sum() * 0.0

            denominator = logits.masked_fill(~non_self, -torch.inf)
            log_probability = logits - torch.logsumexp(
                denominator, dim=1, keepdim=True
            )
            positive_log_probability = log_probability.masked_fill(
                ~positive, 0.0
            )
            mean_positive = (
                positive_log_probability.sum(dim=1)
                / positive_counts.clamp_min(1)
            )
            return -mean_positive[valid].mean()

    def fit_reference(self, data_loader, reference_frame, seed, device=None):
        """Train CDVAE and return the reference latent representation."""
        seed_everything(seed)
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(device)
        epochs = int(self.config["epochs"])
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(CDVAE_OPTIMIZER_WEIGHT_DECAY),
            betas=tuple(CDVAE_OPTIMIZER_BETAS),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(CDVAE_SCHEDULER_ETA_MIN),
        )
        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        warmup_epochs = int(CDVAE_CLUSTER_WARMUP_EPOCHS)
        patience = int(CDVAE_EARLY_STOPPING_PATIENCE)
        best_loss = float("inf")
        best_state = None
        no_improve = 0
        losses = []

        iterator = tqdm(
            range(epochs),
            desc="Training Contrastive-DVAE",
            leave=True,
            dynamic_ncols=True,
        )
        for epoch in iterator:
            self.train()
            epoch_loss = 0.0
            for batch, labels in data_loader:
                batch = batch.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(
                    device_type=device.type, enabled=amp_enabled
                ):
                    reconstructed, mean, log_var, latent = self(batch)
                    reconstruction_loss = F.mse_loss(reconstructed, batch)
                    kl_divergence = -0.5 * torch.mean(
                        torch.sum(
                            1 + log_var - mean.pow(2) - log_var.exp(),
                            dim=1,
                        )
                    )
                    cluster_loss = (
                        self._contrastive_loss(
                            latent, labels, self.config["temperature"]
                        )
                        if epoch >= warmup_epochs
                        else latent.sum() * 0.0
                    )
                    loss = (
                        reconstruction_loss
                        + float(self.config["kl_weight"]) * kl_divergence
                        + float(self.config["cluster_weight"]) * cluster_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    max_norm=float(CDVAE_GRADIENT_CLIP_NORM),
                )
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.item())
                del reconstructed, mean, log_var, latent, loss

            average_loss = epoch_loss / len(data_loader)
            losses.append(average_loss)
            scheduler.step()
            if epoch == warmup_epochs:
                best_loss = float("inf")
                best_state = None
                no_improve = 0
            if average_loss < best_loss:
                best_loss = average_loss
                best_state = deepcopy(self.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    iterator.write(f"Early stopping at epoch {epoch + 1}")
                    break
            iterator.set_postfix(
                loss=f"{average_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

        if best_state is not None:
            self.load_state_dict(best_state)
        self.zero_grad(set_to_none=True)
        del optimizer, scheduler, scaler, best_state

        self.eval()
        reference_tensor = torch.tensor(
            reference_frame.values,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            latent_tensor, _ = self.encoder(reference_tensor)
        latent_values = latent_tensor.cpu().numpy()
        del reference_tensor, latent_tensor
        latent_frame = pd.DataFrame(
            latent_values,
            index=reference_frame.index,
            columns=[
                f"latent_{index}" for index in range(latent_values.shape[1])
            ],
        )
        logger.info("CDVAE training completed")
        return latent_frame, losses

    def encode(self, frame, device=None):
        device = device or next(self.parameters()).device
        batch_size = int(CDVAE_INFERENCE_BATCH_SIZE)
        self.eval()
        encoded = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                batch = torch.as_tensor(
                    frame.iloc[start:start + batch_size].to_numpy(),
                    dtype=torch.float32,
                    device=device,
                )
                mean, _ = self.encoder(batch)
                encoded.append(mean.cpu().numpy())
        return np.concatenate(encoded, axis=0)
