"""Local predictive-coding trainer for stacked belief layers.

The network is trained *without* a global backward graph. Each layer receives
a local error signal and updates its own ``N(mu, var)`` beliefs:

    e_L   = target - output                      (output error)
    e_l   = (e_{l+1} @ W_{l+1}) * f'(z_l)        (local error back-propagation)
    mu   <- mu + alpha * e_l * a_l / var
    var  <- var - gamma * (e_l * a_l)^2 + lambda * var_initial

The ``e_{l+1} @ W_{l+1}`` step only touches adjacent weights, so credit is
assigned locally rather than through one global autograd chain.
"""

from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..core.layers import ProbabilisticBeliefLayer
from ..core.updates import LocalBayesianUpdater


def _tanh(z: torch.Tensor) -> torch.Tensor:
    return torch.tanh(z)


def _tanh_grad(z: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.tanh(z).square()


def _relu(z: torch.Tensor) -> torch.Tensor:
    return F.relu(z)


def _relu_grad(z: torch.Tensor) -> torch.Tensor:
    return (z > 0).to(z.dtype)


def _identity(z: torch.Tensor) -> torch.Tensor:
    return z


def _identity_grad(z: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(z)


_ACTIVATIONS = {
    "tanh": (_tanh, _tanh_grad),
    "relu": (_relu, _relu_grad),
    "identity": (_identity, _identity_grad),
    None: (_identity, _identity_grad),
}


class PredictiveCodingTrainer:
    """Trains a stack of :class:`ProbabilisticBeliefLayer` objects locally.

    Args:
        layers: Ordered sequence of belief layers (input -> output).
        updater: Shared :class:`LocalBayesianUpdater` applied to every layer.
        activation: ``"tanh"`` (default), ``"relu"``, ``"identity"`` or ``None``.
            Applied after every layer except the last, which stays linear.
        threshold: SNR gate used when computing masks / propagating errors.
    """

    def __init__(
        self,
        layers: Sequence[ProbabilisticBeliefLayer],
        updater: LocalBayesianUpdater,
        activation: str | None = "tanh",
        threshold: float = 0.5,
    ) -> None:
        if len(layers) == 0:
            raise ValueError("At least one layer is required.")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; choose from {list(_ACTIVATIONS)}."
            )
        self.layers: List[ProbabilisticBeliefLayer] = list(layers)
        self.updater = updater
        self.activation_name = activation
        self._act, self._act_grad = _ACTIVATIONS[activation]
        self.threshold = threshold

    def _activate(self, z: torch.Tensor, is_last: bool) -> torch.Tensor:
        return z if is_last else self._act(z)

    def _activate_grad(self, z: torch.Tensor, is_last: bool) -> torch.Tensor:
        return torch.ones_like(z) if is_last else self._act_grad(z)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, sample: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return per-layer pre-activations and activations.

        ``activations[0]`` is the input ``x`` and ``activations[-1]`` the
        network output. ``pre_activations[i]`` is the linear output of layer
        ``i`` before the activation function.
        """
        activations = [x]
        pre_activations: List[torch.Tensor] = []
        last = len(self.layers) - 1

        for i, layer in enumerate(self.layers):
            z = layer(activations[-1], threshold=self.threshold, sample=sample)
            pre_activations.append(z)
            activations.append(self._activate(z, is_last=(i == last)))

        return pre_activations, activations

    @torch.no_grad()
    def local_errors(
        self,
        pre_activations: List[torch.Tensor],
        activations: List[torch.Tensor],
        target: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute the local error for every layer (actual minus prediction)."""
        n = len(self.layers)
        errors: List[torch.Tensor | None] = [None] * n
        errors[-1] = target - activations[-1]

        for i in range(n - 2, -1, -1):
            layer = self.layers[i + 1]
            weights = layer.weight_mu * layer.get_sparsity_mask(self.threshold)
            propagated = errors[i + 1] @ weights
            errors[i] = propagated * self._activate_grad(pre_activations[i], is_last=False)

        return errors  # type: ignore[return-value]

    @torch.no_grad()
    def train_step(self, x: torch.Tensor, target: torch.Tensor) -> float:
        """Perform one local predictive-coding update; return the MSE loss."""
        pre_activations, activations = self.forward(x, sample=False)
        output = activations[-1]
        loss = F.mse_loss(output, target)

        errors = self.local_errors(pre_activations, activations, target)
        for i, layer in enumerate(self.layers):
            self.updater.step(layer, activations[i], errors[i])

        return loss.item()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Return the network output for ``x``."""
        _, activations = self.forward(x, sample=sample)
        return activations[-1]

    def compression_rate(self) -> float:
        """Fraction of all pathways pruned across every layer."""
        pruned = 0.0
        total = 0.0
        for layer in self.layers:
            mask = layer.get_sparsity_mask(self.threshold)
            pruned += (1.0 - mask.mean()).item() * mask.numel()
            total += mask.numel()
        return pruned / total
