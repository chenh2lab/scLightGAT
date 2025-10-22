# dvae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from copy import deepcopy

from scLightGAT.logger_config import setup_logger
logger = setup_logger(__name__)

class Encoder(nn.Module):
    """
    Encoder network for DVAE.
    Compresses input data into latent space representations.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.batch_norm2 = nn.BatchNorm1d(512, momentum=0.01)
        self.batch_norm3 = nn.BatchNorm1d(256, momentum=0.01)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.leaky_relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        mean = self.fc_mean(x)
        log_var = self.fc_var(x)
        return mean, log_var

class Decoder(nn.Module):
    """
    Decoder network for DVAE.
    Reconstructs input data from latent space representations.
    """
    def __init__(self, latent_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(256, momentum=0.01)
        self.batch_norm2 = nn.BatchNorm1d(512, momentum=0.01)
        self.batch_norm3 = nn.BatchNorm1d(1024, momentum=0.01)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.leaky_relu(self.batch_norm1(self.fc1(z)))
        z = self.dropout(z)
        z = self.leaky_relu(self.batch_norm2(self.fc2(z)))
        z = self.dropout(z)
        z = self.leaky_relu(self.batch_norm3(self.fc3(z)))
        z = self.dropout(z)
        x_hat = self.fc4(z)
        return x_hat

class DVAE(nn.Module):
    """
    Deep Variational Autoencoder with clustering capability.
    Combines encoding, decoding, and clustering in one model.
    """
    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int, 
                 noise_factor: float = 0.05,
                 cluster_weight: float = 0.5, 
                 temperature: float = 0.1):
        super(DVAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DVAE on device: {self.device}")
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.noise_factor = noise_factor
        self.cluster_weight = cluster_weight
        self.temperature = temperature
        self.to(self.device)
        logger.info(f"DVAE initialized with:")
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Latent dimension: {latent_dim}")
        logger.info(f"Cluster weight: {cluster_weight}")
        logger.info(f"Temperature: {temperature}")

    def get_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x).to(x.device) * self.noise_factor
        return x + noise

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_x = self.add_noise(x)
        mean, log_var = self.encoder(noisy_x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var, z

    def l1_regularization(self, lambda_l1: float) -> torch.Tensor:
        l1_reg = torch.tensor(0., requires_grad=True).to(self.device)
        for param in self.parameters():
            l1_reg = l1_reg + torch.norm(param, p=1)
        return lambda_l1 * l1_reg

    def save_model(self, path: str):
        torch.save({
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'noise_factor': self.noise_factor,
            'cluster_weight': self.cluster_weight,
            'temperature': self.temperature
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state'])
        self.decoder.load_state_dict(checkpoint['decoder_state'])
        self.noise_factor = checkpoint['noise_factor']
        self.cluster_weight = checkpoint['cluster_weight']
        self.temperature = checkpoint['temperature']
        logger.info(f"Model loaded from {path}")