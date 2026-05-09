from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features


def sae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    features: torch.Tensor,
    l1_coeff: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(recon, x)
    l1 = features.abs().mean()
    loss = mse + l1_coeff * l1
    return loss, {"mse": float(mse.detach()), "l1": float(l1.detach()), "loss": float(loss.detach())}

