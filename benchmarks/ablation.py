"""Ablation studies for the Sparse Probabilistic Belief Network.

Two experiments, both on MNIST with the same architecture and seeds:

1. **Variance-rule ablation** — the original error-based rule
   ``var -= gamma * (e * a)^2`` versus the current posterior-precision rule.
2. **Compression sweep** — test accuracy versus the SNR pruning threshold.

Usage (from the project root)::

    python -m benchmarks.ablation
    python -m benchmarks.ablation --epochs 2 --seeds 3
    python -m benchmarks.ablation --epochs 1 --train-subset 10000 --test-subset 2000 --seeds 2
"""

import argparse

import torch

from benchmarks.compare import _summarize, train_spbn
from benchmarks.mnist import _load_mnist, _to_flat_tensors
from spbn_core.core.updates import LocalBayesianUpdater


class ErrorBasedUpdater:
    """The original specification rule: uncertainty shrinks with error.

    ``mu  += alpha * e * a / var``
    ``var -= gamma * (e * a)^2 + lambda * var_initial``

    Kept for ablation only; it makes the network more confident when the error
    is large, which is the behaviour the posterior rule replaced.
    """

    def __init__(
        self,
        learning_rate_mu: float = 1e-3,
        learning_rate_var: float = 1e-3,
        stability_lambda: float = 1e-2,
        var_min: float = 1e-2,
        var_max: float = 5.0,
    ) -> None:
        self.lr_mu = learning_rate_mu
        self.lr_var = learning_rate_var
        self.lambda_reg = stability_lambda
        self.var_min = var_min
        self.var_max = var_max

    @torch.no_grad()
    def step(self, layer, a_prev: torch.Tensor, local_error: torch.Tensor) -> None:
        a_prev = a_prev.detach()
        local_error = local_error.detach()
        batch = a_prev.size(0)

        var = torch.clamp(layer.weight_var.data, min=self.var_min, max=self.var_max)
        precision = 1.0 / var

        hebbian = torch.matmul(local_error.t(), a_prev) / batch
        layer.weight_mu.data.add_(self.lr_mu * hebbian * precision)

        hebbian_sq = torch.matmul(local_error.square().t(), a_prev.square()) / batch
        new_var = var - self.lr_var * hebbian_sq + self.lambda_reg * layer.weight_var_initial
        layer.weight_var.data.copy_(
            torch.clamp(new_var, min=self.var_min, max=self.var_max)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SPBN ablation studies")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sweep-seeds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--lr-mu", type=float, default=1e-3)
    parser.add_argument("--lr-var", type=float, default=5e-2)
    parser.add_argument("--old-lr-var", type=float, default=1e-3)
    parser.add_argument("--stability-lambda", type=float, default=1e-2)
    parser.add_argument("--initial-var", type=float, default=0.05)
    parser.add_argument(
        "--sweep-thresholds", default="0.0,0.1,0.2,0.3", help="comma-separated SNR thresholds"
    )
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    args = parser.parse_args()

    train_set, test_set = _load_mnist(args.data_dir, args.train_subset, args.test_subset)
    x_train, y_train = _to_flat_tensors(train_set)
    x_test, y_test = _to_flat_tensors(test_set)

    common = dict(
        epochs=args.epochs,
        hidden=args.hidden,
        batch_size=args.batch_size,
        initial_var=args.initial_var,
    )

    print("=" * 96)
    print(" SPBN ablation studies on MNIST")
    print("=" * 96)
    print(
        f" train {len(train_set)} | test {len(test_set)} | epochs {args.epochs} "
        f"| seeds {args.seeds} | hidden {args.hidden}"
    )

    print("\n[1] Variance update rule (threshold = %.2f)" % args.threshold)
    print("-" * 96)
    new_rule = [
        train_spbn(
            x_train, y_train, x_test, y_test, threshold=args.threshold, seed=s,
            updater=LocalBayesianUpdater(
                learning_rate_mu=args.lr_mu,
                learning_rate_var=args.lr_var,
                stability_lambda=args.stability_lambda,
            ),
            **common,
        )
        for s in range(args.seeds)
    ]
    old_rule = [
        train_spbn(
            x_train, y_train, x_test, y_test, threshold=args.threshold, seed=s,
            updater=ErrorBasedUpdater(
                learning_rate_mu=args.lr_mu,
                learning_rate_var=args.old_lr_var,
                stability_lambda=args.stability_lambda,
            ),
            **common,
        )
        for s in range(args.seeds)
    ]
    _summarize("posterior precision", new_rule)
    _summarize("error-based (old)", old_rule)

    print("\n[2] Accuracy vs compression (posterior rule)")
    print("-" * 96)
    for threshold in [float(t) for t in args.sweep_thresholds.split(",")]:
        results = [
            train_spbn(
                x_train, y_train, x_test, y_test, threshold=threshold, seed=s,
                updater=LocalBayesianUpdater(
                    learning_rate_mu=args.lr_mu,
                    learning_rate_var=args.lr_var,
                    stability_lambda=args.stability_lambda,
                ),
                **common,
            )
            for s in range(args.sweep_seeds)
        ]
        _summarize(f"threshold {threshold:.2f}", results)

    print("-" * 96)


if __name__ == "__main__":
    main()
