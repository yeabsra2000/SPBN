"""Probabilistic belief layer with reparameterized sampling.

Each synapse is modeled as ``theta_ij ~ N(mu_ij, var_ij)``. During the
forward pass we draw a weight sample using the reparameterization trick and
transmit it through the uncertainty-gated sparsity mask:

    theta_ij = mu_ij + sqrt(var_ij) * eps,   eps ~ N(0, 1)
    z_j      = sum_i theta_ij * M_ij * a_i

A deterministic ("expected") forward pass is also exposed for evaluation and
for local predictive-coding updates, where the sampling noise would otherwise
corrupt the error signal.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparsity import uncertainty_mask


class ProbabilisticBeliefLayer(nn.Module):
    """A linear layer whose weights are normal distributions ``N(mu, var)``.

    Args:
        in_features: Number of input activations.
        out_features: Number of output activations.
        initial_var: Initial belief variance for every synapse.
        snr_threshold: Default SNR gate used by :meth:`get_sparsity_mask`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initial_var: float = 0.01,
        snr_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.snr_threshold = snr_threshold

        # Belief distribution parameters (mu, var) for every synapse.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_var = nn.Parameter(torch.empty(out_features, in_features))

        # Keep the prior variance around so the local updater can regularize
        # the posterior back toward it (lambda * var_initial term).
        self.register_buffer(
            "weight_var_initial",
            torch.full((out_features, in_features), float(initial_var)),
        )

        self.reset_parameters(initial_var)

    def reset_parameters(self, initial_var: float | None = None) -> None:
        """Re-initialize ``mu`` (Kaiming uniform) and ``var`` (constant)."""
        if initial_var is None:
            initial_var = float(self.weight_var_initial.mean().item())
        initial_var = float(initial_var)

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_var, initial_var)
        self.weight_var_initial.fill_(initial_var)

    @property
    def initial_var(self) -> torch.Tensor:
        """Prior variance tensor (used by the local Bayesian updater)."""
        return self.weight_var_initial

    def get_sparsity_mask(self, threshold: float | None = None) -> torch.Tensor:
        """Return the binary SNR mask used to prune uncertain pathways."""
        if threshold is None:
            threshold = self.snr_threshold
        with torch.no_grad():
            return uncertainty_mask(self.weight_mu, self.weight_var, threshold)

    def sample_weights(self) -> torch.Tensor:
        """Draw weights via the reparameterization trick."""
        std = torch.sqrt(torch.clamp(self.weight_var, min=1e-8))
        eps = torch.randn_like(self.weight_mu)
        return self.weight_mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        threshold: float | None = None,
        sample: bool = True,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input activations ``[batch, in_features]``.
            threshold: SNR gate; defaults to ``self.snr_threshold``.
            sample: If ``True`` (default) use stochastic reparameterized
                weights; if ``False`` use the expected weights ``mu``.
            mask: Optional precomputed SNR mask (reused to avoid recomputation).
        """
        if mask is None:
            mask = self.get_sparsity_mask(threshold)
        weights = self.sample_weights() if sample else self.weight_mu
        return F.linear(x, weights * mask)

    def expected_forward(
        self, x: torch.Tensor, threshold: float | None = None
    ) -> torch.Tensor:
        """Deterministic forward pass using ``E[theta] = mu``."""
        return self.forward(x, threshold=threshold, sample=False)

    def sparsity_ratio(self, threshold: float | None = None) -> float:
        """Fraction of pruned pathways (in ``[0, 1]``)."""
        mask = self.get_sparsity_mask(threshold)
        return 1.0 - mask.mean().item()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"snr_threshold={self.snr_threshold}"
        )
