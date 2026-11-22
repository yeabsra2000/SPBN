"""Sparse Probabilistic Belief Network (SPBN).

A small experimental framework that replaces static scalar weights with
per-synapse normal distributions ``N(mu, var)``, updates them with local
Bayesian predictive-coding rules, and prunes uncertain pathways via an
uncertainty-gated (SNR) mask.
"""

from .core.layers import ProbabilisticBeliefLayer
from .core.updates import LocalBayesianUpdater
from .models import SequentialBeliefNetwork
from .training.trainer import PredictiveCodingTrainer

__all__ = [
    "ProbabilisticBeliefLayer",
    "LocalBayesianUpdater",
    "PredictiveCodingTrainer",
    "SequentialBeliefNetwork",
]
