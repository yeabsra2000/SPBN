"""Baseline comparison for the Sparse Probabilistic Belief Network.

Compares three methods on MNIST with the *same* 784 -> hidden -> 10 architecture
and the same MSE-on-one-hot objective, averaged over multiple seeds:

  * SPBN (local Bayesian updates, uncertainty-gated pruning)
  * SPBN without pruning (threshold = 0)
  * MLP trained with Adam + backprop (reference: MSE and cross-entropy)
  * Bayes by Backprop (Blundell et al., 2015), a probabilistic-weight baseline

Usage (from the project root)::

    python -m benchmarks.compare
    python -m benchmarks.compare --epochs 2 --seeds 3
    python -m benchmarks.compare --epochs 1 --train-subset 10000 --test-subset 2000 --seeds 2
"""

import argparse
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.mnist import _load_mnist, _one_hot, _to_flat_tensors
from spbn_core.core.updates import LocalBayesianUpdater
from spbn_core.models import SequentialBeliefNetwork


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class BayesianLinear(nn.Module):
    """Linear layer with a diagonal Gaussian posterior over its weights.

    Implements the Bayes by Backprop reparameterization
    ``w = mu + softplus(rho) * eps`` and exposes its KL divergence to a
    zero-mean Gaussian prior.
    """

    def __init__(self, in_features: int, out_features: int, prior_var: float = 0.1):
        super().__init__()
        self.prior_var = prior_var
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            sigma = F.softplus(self.weight_rho)
            weight = self.weight_mu + sigma * torch.randn_like(self.weight_mu)
        else:
            weight = self.weight_mu
        return F.linear(x, weight, self.bias)

    def kl(self) -> torch.Tensor:
        sigma2 = F.softplus(self.weight_rho).square()
        return 0.5 * torch.sum(
            (self.weight_mu.square() + sigma2) / self.prior_var
            - 1.0
            - torch.log(sigma2 / self.prior_var)
        )


class BayesianMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, prior_var: float):
        super().__init__()
        self.l1 = BayesianLinear(in_features, hidden, prior_var)
        self.l2 = BayesianLinear(hidden, out_features, prior_var)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        return self.l2(torch.tanh(self.l1(x, sample)), sample)

    def kl(self) -> torch.Tensor:
        return self.l1.kl() + self.l2.kl()


@torch.no_grad()
def _accuracy(logits_fn, x: torch.Tensor, labels: torch.Tensor, batch_size: int = 512) -> float:
    correct = 0
    for start in range(0, x.size(0), batch_size):
        xb = x[start : start + batch_size]
        yb = labels[start : start + batch_size]
        correct += (logits_fn(xb).argmax(dim=1) == yb).sum().item()
    return correct / labels.numel()


def train_spbn(
    x_train, y_train, x_test, y_test, *, epochs, hidden, threshold, seed, batch_size=64,
    lr_mu=1e-3, lr_var=5e-2, stability_lambda=1e-2, initial_var=0.05, updater=None,
):
    torch.manual_seed(seed)
    net = SequentialBeliefNetwork(
        x_train.size(1),
        [hidden],
        10,
        initial_var=initial_var,
        snr_threshold=threshold,
        activation="tanh",
    )
    if updater is None:
        updater = LocalBayesianUpdater(
            learning_rate_mu=lr_mu,
            learning_rate_var=lr_var,
            stability_lambda=stability_lambda,
        )
    trainer = net.to_trainer(updater)

    n = x_train.size(0)
    start_time = time.time()
    for _ in range(epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            trainer.train_step(x_train[idx], _one_hot(y_train[idx]))
    elapsed = time.time() - start_time

    acc = _accuracy(lambda xb: trainer.predict(xb), x_test, y_test)
    return acc, net.compression_rate(), _count_params(net), elapsed


def train_mlp(
    x_train, y_train, x_test, y_test, *, epochs, hidden, seed, batch_size=64,
    lr=1e-3, criterion="mse",
):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x_train.size(1), hidden),
        nn.Tanh(),
        nn.Linear(hidden, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if criterion == "mse" else nn.CrossEntropyLoss()
    one_hot_targets = criterion == "mse"

    n = x_train.size(0)
    start_time = time.time()
    for _ in range(epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            target = _one_hot(yb) if one_hot_targets else yb

            optimizer.zero_grad()
            loss = loss_fn(model(xb), target)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start_time

    model.eval()
    acc = _accuracy(model, x_test, y_test)
    return acc, 0.0, _count_params(model), elapsed


def train_bbb(
    x_train, y_train, x_test, y_test, *, epochs, hidden, seed, batch_size=64,
    lr=1e-3, prior_var=0.1, kl_weight=1e-3,
):
    """Bayes by Backprop (Blundell et al., 2015) with an MSE likelihood."""
    torch.manual_seed(seed)
    model = BayesianMLP(x_train.size(1), hidden, 10, prior_var=prior_var)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = x_train.size(0)
    num_batches = (n + batch_size - 1) // batch_size
    start_time = time.time()
    for _ in range(epochs):
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            xb = x_train[idx]
            yb = _one_hot(y_train[idx])

            optimizer.zero_grad()
            nll = F.mse_loss(model(xb, sample=True), yb)
            loss = nll + kl_weight * model.kl() / num_batches
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start_time

    model.eval()
    acc = _accuracy(lambda xb: model(xb, sample=False), x_test, y_test)
    return acc, 0.0, _count_params(model), elapsed


def _summarize(name, results):
    accs = [r[0] for r in results]
    comps = [r[1] for r in results]
    params = results[0][2]
    times = [r[3] for r in results]
    mean = statistics.mean(accs)
    std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    print(
        f" {name:<22} | acc {mean * 100:5.2f}% +/- {std * 100:4.2f} "
        f"| pruned {statistics.mean(comps) * 100:5.1f}% "
        f"| params {params:>7} | {statistics.mean(times):5.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SPBN vs backprop baseline comparison")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--bbb-lr", type=float, default=3e-3)
    parser.add_argument("--bbb-prior-var", type=float, default=1.0)
    parser.add_argument("--bbb-kl-weight", type=float, default=1e-3)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--test-subset", type=int, default=None)
    args = parser.parse_args()

    train_set, test_set = _load_mnist(args.data_dir, args.train_subset, args.test_subset)
    x_train, y_train = _to_flat_tensors(train_set)
    x_test, y_test = _to_flat_tensors(test_set)

    print("=" * 96)
    print(" SPBN vs backprop on MNIST (same architecture, same MSE-on-one-hot objective)")
    print("=" * 96)
    print(
        f" train {len(train_set)} | test {len(test_set)} | epochs {args.epochs} "
        f"| seeds {args.seeds} | hidden {args.hidden} | threshold {args.threshold}"
    )
    print("-" * 96)

    common = dict(epochs=args.epochs, hidden=args.hidden, batch_size=args.batch_size)
    seeds = list(range(args.seeds))

    spbn = [
        train_spbn(x_train, y_train, x_test, y_test, threshold=args.threshold, seed=s, **common)
        for s in seeds
    ]
    spbn_noprune = [
        train_spbn(x_train, y_train, x_test, y_test, threshold=0.0, seed=s, **common)
        for s in seeds
    ]
    mlp_mse = [
        train_mlp(x_train, y_train, x_test, y_test, criterion="mse", seed=s, **common)
        for s in seeds
    ]
    mlp_ce = [
        train_mlp(x_train, y_train, x_test, y_test, criterion="ce", seed=s, **common)
        for s in seeds
    ]
    bbb = [
        train_bbb(
            x_train,
            y_train,
            x_test,
            y_test,
            seed=s,
            lr=args.bbb_lr,
            prior_var=args.bbb_prior_var,
            kl_weight=args.bbb_kl_weight,
            **common,
        )
        for s in seeds
    ]

    _summarize("SPBN (pruned)", spbn)
    _summarize("SPBN (no pruning)", spbn_noprune)
    _summarize("MLP backprop (MSE)", mlp_mse)
    _summarize("MLP backprop (CE)", mlp_ce)
    _summarize("Bayes by Backprop", bbb)
    print("-" * 96)


if __name__ == "__main__":
    main()
