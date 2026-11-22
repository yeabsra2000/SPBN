"""Unit tests for the SPBN core layer and local Bayesian updater.

Run from the project root with::

    python -m unittest discover -s spbn_core/tests -v
"""

import unittest

import torch
import torch.nn.functional as F

from spbn_core.core.layers import ProbabilisticBeliefLayer
from spbn_core.core.sparsity import compression_rate, compute_snr, uncertainty_mask
from spbn_core.core.updates import LocalBayesianUpdater


class TestProbabilisticBeliefLayer(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.in_dim = 8
        self.out_dim = 3
        self.batch = 16
        self.layer = ProbabilisticBeliefLayer(
            self.in_dim, self.out_dim, initial_var=0.1, snr_threshold=0.5
        )
        self.x = torch.randn(self.batch, self.in_dim)

    def test_output_shape(self) -> None:
        out = self.layer(self.x)
        self.assertEqual(tuple(out.shape), (self.batch, self.out_dim))

    def test_parameters_are_belief_distributions(self) -> None:
        self.assertEqual(tuple(self.layer.weight_mu.shape), (self.out_dim, self.in_dim))
        self.assertEqual(tuple(self.layer.weight_var.shape), (self.out_dim, self.in_dim))
        self.assertTrue(torch.all(self.layer.weight_var > 0))

    def test_expected_forward_matches_masked_mu(self) -> None:
        mask = self.layer.get_sparsity_mask()
        manual = F.linear(self.x, self.layer.weight_mu * mask)
        out = self.layer.expected_forward(self.x)
        self.assertTrue(torch.allclose(out, manual, atol=1e-6))

    def test_sampling_is_stochastic(self) -> None:
        self.layer.weight_var.data.fill_(0.5)
        a = self.layer(self.x, threshold=-1.0)
        b = self.layer(self.x, threshold=-1.0)
        self.assertFalse(torch.allclose(a, b))

    def test_sampling_spread_grows_with_variance(self) -> None:
        def spread(var: float) -> float:
            self.layer.weight_var.data.fill_(var)
            samples = torch.stack([self.layer(self.x, threshold=-1.0) for _ in range(32)])
            return samples.std().item()

        self.assertGreater(spread(0.5), spread(0.01))

    def test_zero_variance_is_deterministic(self) -> None:
        def max_difference(var: float) -> float:
            self.layer.weight_var.data.fill_(var)
            a = self.layer(self.x, threshold=-1.0)
            b = self.layer(self.x, threshold=-1.0)
            return (a - b).abs().max().item()

        self.assertLess(max_difference(0.0), max_difference(0.5) * 1e-2)

    def test_mask_is_binary_and_matches_snr(self) -> None:
        mask = self.layer.get_sparsity_mask(threshold=0.5)
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))
        expected = (compute_snr(self.layer.weight_mu, self.layer.weight_var) > 0.5).float()
        self.assertTrue(torch.equal(mask, expected))

    def test_compression_helpers_agree(self) -> None:
        mask = uncertainty_mask(self.layer.weight_mu, self.layer.weight_var, 0.5)
        ratio = compression_rate(self.layer.weight_mu, self.layer.weight_var, 0.5)
        self.assertAlmostEqual(ratio, 1.0 - mask.mean().item(), places=6)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_initial_var_property(self) -> None:
        self.assertTrue(torch.allclose(self.layer.initial_var, torch.full_like(self.layer.initial_var, 0.1)))


class TestLocalBayesianUpdater(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.in_dim = 8
        self.out_dim = 3
        self.batch = 32
        self.threshold = 0.5
        self.layer = ProbabilisticBeliefLayer(
            self.in_dim, self.out_dim, initial_var=0.1, snr_threshold=self.threshold
        )
        self.updater = LocalBayesianUpdater(
            learning_rate_mu=1e-3,
            learning_rate_var=1e-3,
            stability_lambda=1e-3,
        )
        self.x = torch.randn(self.batch, self.in_dim)
        self.target = torch.randn(self.batch, self.out_dim)

    def test_update_keeps_parameters_finite(self) -> None:
        error = self.target - self.layer.expected_forward(self.x, self.threshold)
        self.updater.step(self.layer, self.x, error)
        self.assertTrue(torch.isfinite(self.layer.weight_mu).all())
        self.assertTrue(torch.isfinite(self.layer.weight_var).all())

    def test_variance_respects_bounds(self) -> None:
        for _ in range(20):
            error = self.target - self.layer.expected_forward(self.x, self.threshold)
            self.updater.step(self.layer, self.x, error)
        self.assertGreaterEqual(self.layer.weight_var.min().item(), self.updater.var_min - 1e-9)
        self.assertLessEqual(self.layer.weight_var.max().item(), self.updater.var_max + 1e-9)

    def test_batch_size_mismatch_raises(self) -> None:
        bad_error = torch.randn(self.batch + 1, self.out_dim)
        with self.assertRaises(ValueError):
            self.updater.step(self.layer, self.x, bad_error)

    def test_local_updates_reduce_error(self) -> None:
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(self.batch, self.in_dim, generator=generator)
        w_true = torch.randn(self.out_dim, self.in_dim, generator=generator) / self.in_dim**0.5
        target = x @ w_true.t()

        initial = F.mse_loss(self.layer.expected_forward(x, self.threshold), target).item()
        for _ in range(200):
            prediction = self.layer.expected_forward(x, self.threshold)
            self.updater.step(self.layer, x, target - prediction)
        final = F.mse_loss(self.layer.expected_forward(x, self.threshold), target).item()

        self.assertTrue(torch.isfinite(torch.tensor(final)))
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
