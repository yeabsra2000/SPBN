"""MNIST benchmark for the Sparse Probabilistic Belief Network.

Trains a 784 -> hidden -> 10 belief network with *local* Bayesian updates
(no global backprop) and reports test accuracy, MSE, and pathway compression.

The dataset is downloaded automatically by ``torchvision`` on first run.

Usage (from the project root)::

    python -m benchmarks.mnist
    python -m benchmarks.mnist --epochs 3 --train-subset 5000 --test-subset 1000

If the automatic download fails, see the manual download steps printed in the
error message.
"""

import argparse
import time

import torch
import torch.nn.functional as F

from spbn_core.core.layers import ProbabilisticBeliefLayer
from spbn_core.core.updates import LocalBayesianUpdater
from spbn_core.training.trainer import PredictiveCodingTrainer

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _load_mnist(data_dir: str, train_subset: int | None, test_subset: int | None):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "torchvision is required for the MNIST benchmark.\n"
            "Install it with:  pip install torchvision"
        ) from exc

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    try:
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    except Exception as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Could not download/prepare MNIST.\n"
            f"Reason: {exc}\n\n"
            "Manual option: download these four files into "
            f"{data_dir}/MNIST/raw/ from {MNIST_MIRROR}:\n"
            "  train-images-idx3-ubyte.gz\n"
            "  train-labels-idx1-ubyte.gz\n"
            "  t10k-images-idx3-ubyte.gz\n"
            "  t10k-labels-idx1-ubyte.gz\n"
            "then re-run this script."
        ) from exc

    if train_subset is not None:
        train_set = torch.utils.data.Subset(train_set, range(min(train_subset, len(train_set))))
    if test_subset is not None:
        test_set = torch.utils.data.Subset(test_set, range(min(test_subset, len(test_set))))

    return train_set, test_set


def _to_flat_tensors(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    return images.reshape(images.size(0), -1), labels


def _one_hot(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


@torch.no_grad()
def evaluate(
    trainer: PredictiveCodingTrainer,
    x: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 512,
) -> tuple[float, float]:
    correct = 0
    total = 0
    mse_sum = 0.0

    for start in range(0, x.size(0), batch_size):
        xb = x[start : start + batch_size]
        yb = labels[start : start + batch_size]
        logits = trainer.predict(xb)
        mse_sum += F.mse_loss(logits, _one_hot(yb), reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.numel()

    return correct / total, mse_sum / total


def main() -> None:
    parser = argparse.ArgumentParser(description="SPBN MNIST benchmark")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr-mu", type=float, default=1e-3)
    parser.add_argument("--lr-var", type=float, default=5e-2)
    parser.add_argument("--stability-lambda", type=float, default=1e-2)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--initial-var", type=float, default=0.05)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_set, test_set = _load_mnist(args.data_dir, args.train_subset, args.test_subset)
    x_train, y_train = _to_flat_tensors(train_set)
    x_test, y_test = _to_flat_tensors(test_set)

    in_dim = x_train.size(1)
    layers = [
        ProbabilisticBeliefLayer(
            in_dim, args.hidden, initial_var=args.initial_var, snr_threshold=args.threshold
        ),
        ProbabilisticBeliefLayer(
            args.hidden, 10, initial_var=args.initial_var, snr_threshold=args.threshold
        ),
    ]
    updater = LocalBayesianUpdater(
        learning_rate_mu=args.lr_mu,
        learning_rate_var=args.lr_var,
        stability_lambda=args.stability_lambda,
    )
    trainer = PredictiveCodingTrainer(
        layers, updater, activation="tanh", threshold=args.threshold
    )

    print("=" * 64)
    print(" SPBN MNIST benchmark (local Bayesian updates, no backprop)")
    print("=" * 64)
    print(f" train samples: {len(train_set):>6} | test samples: {len(test_set):>6}")
    print(f" architecture : {in_dim} -> {args.hidden} -> 10")
    print(f" epochs: {args.epochs} | batch: {args.batch_size} | threshold: {args.threshold}")

    n = x_train.size(0)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n, args.batch_size):
            idx = permutation[start : start + args.batch_size]
            xb = x_train[idx]
            yb = _one_hot(y_train[idx])
            epoch_loss += trainer.train_step(xb, yb)
            num_batches += 1

        train_acc, _ = evaluate(trainer, x_train, y_train)
        test_acc, test_mse = evaluate(trainer, x_test, y_test)
        elapsed = time.time() - start_time

        print(
            f" epoch {epoch:2d}/{args.epochs} | loss {epoch_loss / num_batches:.4f} "
            f"| train acc {train_acc * 100:5.2f}% | test acc {test_acc * 100:5.2f}% "
            f"| test mse {test_mse:.4f} | pruned {trainer.compression_rate() * 100:5.2f}% "
            f"| {elapsed:5.1f}s"
        )

    print("-" * 64)
    print(f" final test accuracy : {test_acc * 100:.2f}%")
    print(f" final compression   : {trainer.compression_rate() * 100:.2f}% pruned pathways")


if __name__ == "__main__":
    main()
