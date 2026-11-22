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
└── compare.py             # SPBN vs backprop baseline comparison

main.py                    # Entry point for running toy benchmarks (e.g., MNIST/Synthetic)
```

## Quickstart
```bash
pip install torch torchvision

python main.py                                          # run training demos
python -m unittest discover -s spbn_core/tests -v       # run the test suite
python -m benchmarks.mnist                              # MNIST benchmark (auto-downloads data)
python -m benchmarks.compare                            # SPBN vs backprop baselines
```

The MNIST benchmark downloads the dataset into `./data/` on first run. If the
automatic download fails, download the four `*-ubyte.gz` files from
`https://ossci-datasets.s3.amazonaws.com/mnist/` into `data/MNIST/raw/` and re-run.

## Baseline Results (MNIST)

Same 784 -> 128 -> 10 architecture and MSE-on-one-hot objective, 3 seeds, 3 epochs:

| Method | Test accuracy | Pruned | Params | Time |
|---|---|---|---|---|
| SPBN (pruned, threshold 0.1) | 94.13% ± 0.30 | 57.9% | 203,264 | 13.8s |
| SPBN (no pruning) | 94.72% ± 0.07 | 0.0% | 203,264 | 13.8s |
| MLP backprop (MSE) | 96.49% ± 0.12 | 0.0% | 101,770 | 6.1s |
| MLP backprop (cross-entropy) | 96.88% ± 0.14 | 0.0% | 101,770 | 5.6s |

**Takeaway:** local Bayesian updates land ~2.4–2.8 points behind backprop at
this scale, pruning costs ~0.6 points, and the belief network uses ~2x the
parameters and ~2.3x wall-clock. This is a functional research prototype, not
a drop-in replacement for backprop.

## Implementation Status
- [x] Probabilistic belief layer with reparameterized sampling and SNR gating
- [x] Local Bayesian belief updater (no global backprop / no optimizer)
- [x] Uncertainty-gated sparsity and compression helpers
- [x] Multi-layer local predictive-coding trainer
- [x] Unit tests and single/multi-layer training demos
- [x] MNIST benchmark (~94.1% ± 0.3 with ~58% pathways pruned)
- [x] Multi-seed baseline comparison against backprop
