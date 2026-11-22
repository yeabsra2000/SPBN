"""Local Bayesian belief-updating mechanics.

Instead of a global ``loss.backward()`` chain, each layer adjusts its own
``N(mu, var)`` beliefs from a *local* prediction error.

Mean update (precision-weighted delta rule), with ``e_j`` the local error on
output ``j`` and ``a_i`` the incoming activation on input ``i``::

    mu_ij <- mu_ij + alpha * e_j * a_i / var_ij

Variance update (Bayesian posterior precision). Each active input contributes
evidence that sharpens the belief, so the posterior precision is the prior
precision plus the accumulated input energy::

    precision_ij <- 1 / var_ij + gamma * mean_b(a_i^2)
    var_ij       <- 1 / precision_ij
    var_ij       <- var_ij + lambda * (var_initial - var_ij)

The final term pulls the variance back toward the prior, preventing collapse.
Unlike the earlier ``var -= gamma * (e * a)^2`` rule, this does *not* make the
network more confident precisely when it is wrong; error magnitude affects the
mean, while input evidence controls uncertainty.

The error sign convention is "actual minus prediction"
(``e = target - prediction``), so the additive ``mu`` term moves the belief
toward the target. Updates are averaged over the batch dimension.
"""

import torch


class LocalBayesianUpdater:
    """Applies local Bayesian belief updates to a belief layer.

    Args:
        learning_rate_mu: Step size ``alpha`` for the mean update.
        learning_rate_var: Weight ``gamma`` of the input-evidence term that
            sharpens the posterior precision.
        stability_lambda: Weight ``lambda`` of the prior pull-back term that
            keeps the variance from collapsing.
        var_min: Lower clamp for the variance (numerical stability).
        var_max: Upper clamp for the variance.
    """

    def __init__(
        self,
        learning_rate_mu: float = 1e-3,
        learning_rate_var: float = 5e-2,
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
    def step(
        self,
        layer,
        a_prev: torch.Tensor,
        local_error: torch.Tensor,
    ) -> None:
        """Update ``layer``'s belief parameters in place.

        Args:
            layer: A :class:`ProbabilisticBeliefLayer` instance.
            a_prev: Incoming activations ``[batch, in_features]``.
            local_error: Local prediction error ``e = target - prediction``,
                shaped ``[batch, out_features]``.
        """
        a_prev = a_prev.detach()
        local_error = local_error.detach()

        if a_prev.dim() != 2 or local_error.dim() != 2:
            raise ValueError("a_prev and local_error must be 2-D tensors.")
        if a_prev.size(0) != local_error.size(0):
            raise ValueError("a_prev and local_error must share the batch size.")

        batch = a_prev.size(0)

        # Current precision (inverse variance), clamped for stability.
        var = torch.clamp(layer.weight_var.data, min=self.var_min, max=self.var_max)
        precision = 1.0 / var

        # Bayesian posterior precision: prior precision + input evidence.
        # a_prev^2 averaged over the batch is the per-input evidence [1, in].
        evidence = a_prev.square().mean(dim=0, keepdim=True)
        posterior_var = 1.0 / (precision + self.lr_var * evidence)

        # Pull the posterior variance back toward the prior to avoid collapse.
        var_initial = layer.weight_var_initial
        posterior_var = posterior_var + self.lambda_reg * (var_initial - posterior_var)
        posterior_var = torch.clamp(posterior_var, min=self.var_min, max=self.var_max)
        posterior_precision = 1.0 / posterior_var

        # Precision-weighted local delta rule: mean_b(e_j * a_i).
        hebbian = torch.matmul(local_error.t(), a_prev) / batch
        layer.weight_mu.data.add_(self.lr_mu * hebbian * posterior_precision)

        layer.weight_var.data.copy_(posterior_var)
