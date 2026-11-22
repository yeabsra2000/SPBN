# Sparse Probabilistic Belief Network (SPBN)

## Project Vision
An experimental neuromorphic deep learning framework that replaces traditional global backpropagation and static floating-point weights with **local Bayesian belief-updating distributions** ($\mu, \sigma^2$), **predictive coding error correction**, and **event-driven structural sparsity** (dynamic uncertainty-gated pruning).

## Core Architectural Pillars
1. **Probabilistic Synapses:** Every connection parameter is modeled as a normal distribution $\mathcal{N}(\mu, \sigma^2)$ rather than a static scalar.
2. **Local Predictive Coding:** Layers update their belief states locally using prediction errors ($e_j = a_j - \mathbb{E}[a_j]$) instead of global backward chains.
3. **Uncertainty-Gated Sparsity (Implicit Compression):** Pathways are dynamically masked out based on their Signal-to-Noise Ratio (SNR), dropping idle or uncertain memory footprints to zero.

## Directory Structure
```text
spbn_core/
│
├── models.py              # SequentialBeliefNetwork container + save/load
├── core/
│   ├── __init__.py
│   ├── layers.py          # ProbabilisticBeliefLayer with reparameterization
│   ├── sparsity.py        # Dynamic uncertainty mask & compression gates
│   └── updates.py         # Local Bayesian belief-updating mechanics
│
├── training/
│   ├── __init__.py
│   └── trainer.py         # Non-backprop local predictive optimization loop
│
├── tests/
│   ├── test_layer.py      # Unit tests for tensor shapes, sampling, and local updates
│   ├── test_trainer.py    # Unit tests for multi-layer local training
│   └── test_network.py    # Unit tests for the sequential model + save/load
│
└── __init__.py

benchmarks/
├── mnist.py               # MNIST benchmark (auto-downloads the dataset)
├── compare.py             # SPBN vs backprop baseline comparison
└── ablation.py            # variance-rule and compression ablations

main.py                    # Entry point for running toy benchmarks (e.g., MNIST/Synthetic)
```

## Quickstart
```bash
pip install torch torchvision

python main.py                                          # run training demos
python -m unittest discover -s spbn_core/tests -v       # run the test suite
python -m benchmarks.mnist                              # MNIST benchmark (auto-downloads data)
python -m benchmarks.compare                            # SPBN vs backprop baselines
python -m benchmarks.ablation                           # variance + compression ablations
```

The MNIST benchmark downloads the dataset into `./data/` on first run. If the
automatic download fails, download the four `*-ubyte.gz` files from
`https://ossci-datasets.s3.amazonaws.com/mnist/` into `data/MNIST/raw/` and re-run.

## Baseline Results (MNIST)

Same 784 -> 128 -> 10 architecture and MSE-on-one-hot objective, 3 seeds, 3 epochs:

| Method | Test accuracy | Pruned | Params | Time |
|---|---|---|---|---|
| SPBN (pruned, threshold 0.1) | 94.13% ± 0.30 | 57.9% | 203,264 | 13.2s |
| SPBN (no pruning) | 94.72% ± 0.07 | 0.0% | 203,264 | 13.6s |
| MLP backprop (MSE) | 96.49% ± 0.12 | 0.0% | 101,770 | 11.3s |
| MLP backprop (cross-entropy) | 96.88% ± 0.14 | 0.0% | 101,770 | 9.7s |
| Bayes by Backprop (MSE) | 93.88% ± 0.18 | 0.0% | 203,402 | 34.4s |

**Takeaway:** SPBN is competitive with Bayes by Backprop — the closest
probabilistic-weight baseline — while pruning 58% of pathways and running
~2.6x faster. It remains ~2.4–2.8 points behind deterministic backprop at this
scale. This is a functional research prototype, not a drop-in replacement for
backprop. (Bayes by Backprop was tuned lightly on a subset; treat its number as
indicative.)

## Ablations (MNIST)

### Variance update rule (threshold 0.1, 3 seeds, 3 epochs)

| Rule | Test accuracy | Pruned |
|---|---|---|
| Posterior precision (current) | 94.13% ± 0.30 | 57.9% |
| Error-based (original specification) | 35.58% ± 3.82 | 100.0% |

The original rule *shrinks* variance when the error is large. Over training the
variance collapses, the SNR gate prunes every pathway (100%), and the network
dies. The posterior-precision rule avoids this collapse — this is the single
most important correctness fix in the project.

### Accuracy vs compression (posterior rule, 2 seeds)

| SNR threshold | Test accuracy | Pruned |
|---|---|---|
| 0.00 | 94.67% ± 0.02 | 0.0% |
| 0.10 | 94.10% ± 0.36 | 57.9% |
| 0.20 | 90.34% ± 0.00 | 94.7% |
| 0.30 | 88.09% ± 0.82 | 98.0% |

Pruning ~58% of pathways costs only ~0.6 points; aggressive pruning (>90%)
costs several points.

## Implementation Status
- [x] Probabilistic belief layer with reparameterized sampling and SNR gating
- [x] Local Bayesian belief updater (no global backprop / no optimizer)
- [x] Uncertainty-gated sparsity and compression helpers
- [x] Multi-layer local predictive-coding trainer
- [x] Unit tests and single/multi-layer training demos
- [x] MNIST benchmark (~94.1% ± 0.3 with ~58% pathways pruned)
- [x] Multi-seed baseline comparison vs backprop and Bayes by Backprop
- [x] Variance-rule and compression ablations
