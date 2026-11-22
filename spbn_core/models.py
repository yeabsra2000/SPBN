"""A sequential container for belief layers.

Wraps a stack of :class:`ProbabilisticBeliefLayer` objects with an activation,
a forward pass for inference, save/load helpers, and a factory that returns a
:class:`PredictiveCodingTrainer` for local (no-backprop) training.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.layers import ProbabilisticBeliefLayer
from .core.updates import LocalBayesianUpdater
from .training.trainer import PredictiveCodingTrainer

_VALID_ACTIVATIONS = {"tanh", "relu", "identity", None}


def _apply_activation(z: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in (None, "identity"):
        return z
    if activation == "tanh":
        return torch.tanh(z)
    if activation == "relu":
        return F.relu(z)
    raise ValueError(f"Unknown activation {activation!r}.")


class SequentialBeliefNetwork(nn.Module):
    """A feed-forward stack of probabilistic belief layers.

    Args:
        in_features: Input dimensionality.
        hidden_sizes: Hidden layer widths, e.g. ``[128, 64]``.
        out_features: Output dimensionality.
        initial_var: Initial belief variance for every synapse.
        snr_threshold: SNR gate used by every layer.
        activation: Activation after each hidden layer (last layer is linear).
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int],
        out_features: int,
        initial_var: float = 0.01,
        snr_threshold: float = 0.05,
        activation: str | None = "tanh",
    ) -> None:
        super().__init__()
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; "
                f"choose from {sorted(a for a in _VALID_ACTIVATIONS if a)}."
            )

        sizes: List[int] = [in_features, *hidden_sizes, out_features]
        self.layers = nn.ModuleList(
            [
                ProbabilisticBeliefLayer(
                    sizes[i],
                    sizes[i + 1],
                    initial_var=initial_var,
                    snr_threshold=snr_threshold,
                )
                for i in range(len(sizes) - 1)
            ]
        )
        self.activation = activation
        self.snr_threshold = snr_threshold

        self.config = {
            "in_features": in_features,
            "hidden_sizes": list(hidden_sizes),
            "out_features": out_features,
            "initial_var": initial_var,
            "snr_threshold": snr_threshold,
            "activation": activation,
        }

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Inference forward pass; ``sample=True`` draws stochastic weights."""
        last = len(self.layers) - 1
        a = x
        for i, layer in enumerate(self.layers):
            z = layer(a, threshold=self.snr_threshold, sample=sample)
            a = z if i == last else _apply_activation(z, self.activation)
        return a

    def expected_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic forward pass using expected weights."""
        return self.forward(x, sample=False)

    def to_trainer(self, updater: LocalBayesianUpdater) -> PredictiveCodingTrainer:
        """Build a local predictive-coding trainer over this network's layers."""
        return PredictiveCodingTrainer(
            self.layers,
            updater,
            activation=self.activation,
            threshold=self.snr_threshold,
        )

    def compression_rate(self) -> float:
        """Fraction of all pathways pruned across every layer."""
        pruned = 0.0
        total = 0.0
        for layer in self.layers:
            mask = layer.get_sparsity_mask(self.snr_threshold)
            pruned += (1.0 - mask.mean()).item() * mask.numel()
            total += mask.numel()
        return pruned / total

    def save(self, path: str) -> None:
        """Save the model configuration and parameters to ``path``."""
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "SequentialBeliefNetwork":
        """Load a model previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
