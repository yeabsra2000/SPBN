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
│   └── test_trainer.py    # Unit tests for multi-layer local training
│
└── __init__.py

benchmarks/
└── mnist.py               # MNIST benchmark (auto-downloads the dataset)

main.py                    # Entry point for running toy benchmarks (e.g., MNIST/Synthetic)
```

## Quickstart
```bash
pip install torch torchvision

python main.py                                          # run training demos
python -m unittest discover -s spbn_core/tests -v       # run the test suite
python -m benchmarks.mnist                              # MNIST benchmark (auto-downloads data)
```

The MNIST benchmark downloads the dataset into `./data/` on first run. If the
automatic download fails, download the four `*-ubyte.gz` files from
`https://ossci-datasets.s3.amazonaws.com/mnist/` into `data/MNIST/raw/` and re-run.

## Implementation Status
- [x] Probabilistic belief layer with reparameterized sampling and SNR gating
- [x] Local Bayesian belief updater (no global backprop / no optimizer)
- [x] Uncertainty-gated sparsity and compression helpers
- [x] Multi-layer local predictive-coding trainer
- [x] Unit tests and single/multi-layer training demos
- [x] MNIST benchmark (~93.6% test accuracy with ~58% pathways pruned)
