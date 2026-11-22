"""Unit tests for the multi-layer local predictive-coding trainer."""

import unittest

import torch
import torch.nn.functional as F

from spbn_core.core.layers import ProbabilisticBeliefLayer
from spbn_core.core.updates import LocalBayesianUpdater
from spbn_core.training.trainer import PredictiveCodingTrainer


def _make_teacher(batch: int, in_dim: int, hidden: int, out_dim: int, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, in_dim, generator=generator)
    w1 = torch.randn(hidden, in_dim, generator=generator) / in_dim**0.5
    w2 = torch.randn(out_dim, hidden, generator=generator) / hidden**0.5
    target = torch.tanh(x @ w1.t()) @ w2.t()
    return x, target


class TestPredictiveCodingTrainer(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch = 32
        self.in_dim = 8
        self.hidden = 16
        self.out_dim = 3
        self.threshold = 0.5

        self.layers = [
            ProbabilisticBeliefLayer(
                self.in_dim, self.hidden, initial_var=0.05, snr_threshold=self.threshold
            ),
            ProbabilisticBeliefLayer(
                self.hidden, self.out_dim, initial_var=0.05, snr_threshold=self.threshold
            ),
        ]
        self.updater = LocalBayesianUpdater(
            learning_rate_mu=2e-3, learning_rate_var=1e-3, stability_lambda=1e-3
        )
        self.trainer = PredictiveCodingTrainer(
            self.layers, self.updater, activation="tanh", threshold=self.threshold
        )
        self.x = torch.randn(self.batch, self.in_dim)
        self.target = torch.randn(self.batch, self.out_dim)

    def test_forward_shapes(self) -> None:
        pre, acts = self.trainer.forward(self.x)
        self.assertEqual(len(pre), 2)
        self.assertEqual(len(acts), 3)
        self.assertEqual(tuple(acts[0].shape), (self.batch, self.in_dim))
        self.assertEqual(tuple(acts[1].shape), (self.batch, self.hidden))
        self.assertEqual(tuple(acts[2].shape), (self.batch, self.out_dim))

    def test_local_error_shapes(self) -> None:
        pre, acts = self.trainer.forward(self.x)
        errors = self.trainer.local_errors(pre, acts, self.target)
        self.assertEqual(len(errors), 2)
        self.assertEqual(tuple(errors[0].shape), (self.batch, self.hidden))
        self.assertEqual(tuple(errors[1].shape), (self.batch, self.out_dim))

    def test_compression_rate_in_range(self) -> None:
        rate = self.trainer.compression_rate()
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_invalid_activation_raises(self) -> None:
        with self.assertRaises(ValueError):
            PredictiveCodingTrainer(self.layers, self.updater, activation="bogus")

    def test_train_step_reduces_loss(self) -> None:
        x, target = _make_teacher(self.batch, self.in_dim, self.hidden, self.out_dim)
        initial = F.mse_loss(self.trainer.predict(x), target).item()
        for _ in range(300):
            self.trainer.train_step(x, target)
        final = F.mse_loss(self.trainer.predict(x), target).item()

        self.assertTrue(torch.isfinite(torch.tensor(final)))
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
