"""Local Bayesian belief-updating mechanics.

Instead of a global ``loss.backward()`` chain, each layer adjusts its own
``N(mu, var)`` beliefs from a *local* prediction error. Following the SPBN
specifications, with ``e_j`` the local error on output ``j`` and ``a_i`` the
incoming activation on input ``i``:

    mu_ij      <- mu_ij + alpha * e_j * a_i / var_ij
    var_ij     <- var_ij - gamma * (e_j * a_i)^2 + lambda * var_initial

The error sign convention is "actual minus prediction"
(``e = target - prediction``), so the additive ``mu`` term moves the belief
toward the target. Updates are averaged over the batch dimension.
"""

import torch


class LocalBayesianUpdater:
    """Applies local Bayesian belief updates to a belief layer.

    Args:
        learning_rate_mu: Step size ``alpha`` for the mean update.
        learning_rate_var: Step size ``gamma`` for the variance update.
        stability_lambda: Weight ``lambda`` of the prior pull-back term that
            keeps the variance from collapsing.
        var_min: Lower clamp for the variance (numerical stability).
        var_max: Upper clamp for the variance.
    """

    def __init__(
        self,
        learning_rate_mu: float = 1e-3,
        learning_rate_var: float = 1e-3,
        stability_lambda: float = 1e-3,
        var_min: float = 0.05,
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
        var = torch.clamp(layer.weight_var.data, min=self.var_min)
        precision = 1.0 / var

        # mean_b(e_j * a_i)  -> [out_features, in_features]
        hebbian = torch.matmul(local_error.t(), a_prev) / batch

        # mu_new = mu + alpha * e_j * a_i / var
        layer.weight_mu.data.add_(self.lr_mu * hebbian * precision)

        # mean_b((e_j * a_i)^2) -> [out_features, in_features]
        hebbian_sq = torch.matmul(local_error.square().t(), a_prev.square()) / batch

        # var_new = var - gamma * (e_j * a_i)^2 + lambda * var_initial
        var_initial = layer.weight_var_initial
        new_var = var - self.lr_var * hebbian_sq + self.lambda_reg * var_initial
        layer.weight_var.data.copy_(
            torch.clamp(new_var, min=self.var_min, max=self.var_max)
        )
