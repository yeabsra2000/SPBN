"""Uncertainty-gated structural sparsity.

Every synapse is a normal distribution ``N(mu, var)``. A synapse is only
allowed to transmit when its signal-to-noise ratio (SNR) is high enough:

    SNR_ij = |mu_ij| / (sqrt(var_ij) + eps)
    M_ij   = I(SNR_ij > threshold)

Pruned pathways (``M_ij == 0``) are removed from the effective weight matrix,
which gives the network an implicit, dynamic compression mechanism.
"""

import torch

EPS = 1e-8


def compute_snr(
    weight_mu: torch.Tensor,
    weight_var: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Return the per-synapse signal-to-noise ratio."""
    std = torch.sqrt(torch.clamp(weight_var, min=eps))
    return torch.abs(weight_mu) / (std + eps)


def uncertainty_mask(
    weight_mu: torch.Tensor,
    weight_var: torch.Tensor,
    threshold: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Return a binary mask ``M`` where confident pathways are kept."""
    snr = compute_snr(weight_mu, weight_var, eps=eps)
    return (snr > threshold).to(weight_mu.dtype)


def compression_rate(
    weight_mu: torch.Tensor,
    weight_var: torch.Tensor,
    threshold: float = 1.0,
) -> float:
    """Fraction of pathways that are pruned (in ``[0, 1]``)."""
    mask = uncertainty_mask(weight_mu, weight_var, threshold=threshold)
    return 1.0 - mask.mean().item()
