"""Functional training test for the Sparse Probabilistic Belief Network.

This script trains a :class:`ProbabilisticBeliefLayer` to fit a synthetic
linear task using *only* local Bayesian belief updates -- no ``loss.backward``
and no optimizer. It verifies that the local predictive-coding loop reduces
the prediction error, demonstrates stochastic reparameterized sampling, and
reports the uncertainty-gated compression rate.

Run from the project root with::

    python main.py
"""

import torch
import torch.nn.functional as F

from spbn_core.core.layers import ProbabilisticBeliefLayer
from spbn_core.core.updates import LocalBayesianUpdater


def make_linear_task(
    batch_size: int,
    in_dim: int,
    out_dim: int,
    noise: float = 0.0,
    seed: int = 0,
):
    """Generate a synthetic regression task ``target = x @ W_true.T + noise``."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, in_dim, generator=generator)
    w_true = torch.randn(out_dim, in_dim, generator=generator) / in_dim**0.5
    target = x @ w_true.t()
    if noise > 0.0:
        target = target + noise * torch.randn(target.shape, generator=generator)
    return x, target, w_true


@torch.no_grad()
def mse(layer: ProbabilisticBeliefLayer, x: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    """Deterministic (expected-weight) mean squared error."""
    prediction = layer.expected_forward(x, threshold=threshold)
    return F.mse_loss(prediction, target).item()


@torch.no_grad()
def train(
    layer: ProbabilisticBeliefLayer,
    updater: LocalBayesianUpdater,
    x: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    steps: int,
    log_every: int = 50,
):
    """Run the non-backprop local predictive-coding optimization loop."""
    history = []
    for step in range(1, steps + 1):
        # Local prediction error uses the "actual minus prediction" convention
        # required by the local Bayesian update rule: e = target - prediction.
        prediction = layer.expected_forward(x, threshold=threshold)
        local_error = target - prediction
        loss = F.mse_loss(prediction, target)

        updater.step(layer, x, local_error)

        history.append(loss.item())
        if step == 1 or step % log_every == 0:
            print(f"  step {step:4d} | mse = {loss.item():.6f}")

    return history


@torch.no_grad()
def sampling_report(layer: ProbabilisticBeliefLayer, x: torch.Tensor, threshold: float) -> None:
    """Show that forward passes are stochastic (sampled synaptic weights)."""
    samples = torch.stack(
        [layer(x, threshold=threshold, sample=True) for _ in range(64)]
    )
    print(f"  sample output mean: {samples.mean().item():+.4f}")
    print(f"  sample output std : {samples.std().item():.4f} (stochastic synapses)")


def main() -> None:
    print("=" * 62)
    print(" Sparse Probabilistic Belief Network -- local training test")
    print("=" * 62)

    # ------------------------------------------------------------------ config
    torch.manual_seed(0)

    batch_size = 64
    in_dim = 32
    out_dim = 8
    steps = 400
    threshold = 0.5

    layer = ProbabilisticBeliefLayer(
        in_features=in_dim,
        out_features=out_dim,
        initial_var=0.1,
        snr_threshold=threshold,
    )
    updater = LocalBayesianUpdater(
        learning_rate_mu=1e-3,
        learning_rate_var=1e-3,
        stability_lambda=1e-3,
    )

    # ------------------------------------------------------------- synthetic task
    x, target, _ = make_linear_task(batch_size, in_dim, out_dim, noise=0.0, seed=1)

    initial_loss = mse(layer, x, target, threshold)
    initial_sparsity = layer.sparsity_ratio(threshold)
    print(f"\nInitial mse        : {initial_loss:.6f}")
    print(f"Initial compression: {initial_sparsity * 100:.2f}% pruned")

    # --------------------------------------------------- local belief optimization
    print("\nTraining with local Bayesian belief updates (no backprop)...")
    history = train(layer, updater, x, target, threshold, steps)

    final_loss = mse(layer, x, target, threshold)
    final_sparsity = layer.sparsity_ratio(threshold)

    # ------------------------------------------------------------------- report
    print("\nResults")
    print(f"  initial mse     : {initial_loss:.6f}")
    print(f"  final mse       : {final_loss:.6f}")
    print(f"  error reduction : {(1.0 - final_loss / initial_loss) * 100:.2f}%")
    print(f"  final compression: {final_sparsity * 100:.2f}% pruned pathways")

    print("\nStochastic forward-pass sampling check")
    sampling_report(layer, x, threshold)

    # --------------------------------------------------------------- assertions
    assert torch.isfinite(torch.tensor(final_loss)), "loss became non-finite"
    assert final_loss < initial_loss, (
        f"local updates failed to reduce error "
        f"({initial_loss:.6f} -> {final_loss:.6f})"
    )
    assert 0.0 <= final_sparsity <= 1.0, "invalid sparsity ratio"

    print("\nAll checks passed: the local Bayesian pipeline is functional.")


if __name__ == "__main__":
    main()
