"""Unit tests for the sequential belief network container."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from spbn_core.core.updates import LocalBayesianUpdater
from spbn_core.models import SequentialBeliefNetwork


def _make_teacher(batch: int, in_dim: int, hidden: int, out_dim: int, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, in_dim, generator=generator)
    w1 = torch.randn(hidden, in_dim, generator=generator) / in_dim**0.5
    w2 = torch.randn(out_dim, hidden, generator=generator) / hidden**0.5
    return x, torch.tanh(x @ w1.t()) @ w2.t()


class TestSequentialBeliefNetwork(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch = 32
        self.in_dim = 8
        self.hidden = 16
        self.out_dim = 3
        self.net = SequentialBeliefNetwork(
            self.in_dim,
            [self.hidden],
            self.out_dim,
            initial_var=0.05,
            snr_threshold=0.1,
            activation="tanh",
        )
        self.x = torch.randn(self.batch, self.in_dim)

    def test_forward_shape(self) -> None:
        self.assertEqual(tuple(self.net(self.x).shape), (self.batch, self.out_dim))

    def test_expected_forward_is_deterministic(self) -> None:
        a = self.net.expected_forward(self.x)
        b = self.net.expected_forward(self.x)
        self.assertTrue(torch.allclose(a, b))

    def test_compression_rate_in_range(self) -> None:
        rate = self.net.compression_rate()
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_invalid_activation_raises(self) -> None:
        with self.assertRaises(ValueError):
            SequentialBeliefNetwork(4, [8], 2, activation="bogus")

    def test_save_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            self.net.save(path)
            loaded = SequentialBeliefNetwork.load(path)

        self.assertEqual(loaded.config, self.net.config)
        self.assertTrue(
            torch.allclose(loaded.expected_forward(self.x), self.net.expected_forward(self.x))
        )

    def test_to_trainer_reduces_loss(self) -> None:
        x, target = _make_teacher(self.batch, self.in_dim, self.hidden, self.out_dim)
        trainer = self.net.to_trainer(LocalBayesianUpdater())
        initial = F.mse_loss(trainer.predict(x), target).item()
        for _ in range(200):
            trainer.train_step(x, target)
        final = F.mse_loss(trainer.predict(x), target).item()

        self.assertTrue(torch.isfinite(torch.tensor(final)))
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
