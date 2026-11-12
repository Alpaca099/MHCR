"""Coverage-calibrated prefix selection.

A query is ranked against the cell prototypes, then the number of cells to
scan is chosen per query as the prefix that maximises

    covered_probability(prefix) - penalty * scanned_vectors(prefix)

There is no upper bound on the prefix length: the prefix ends where the next
cell stops paying for itself.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["ACRAdaptive", "prefix_budget"]


def prefix_budget(predcov, scans, penalty):
    """Per-query prefix length r(q) = argmax_r [sum p - penalty * sum c], at least 1."""
    predcov = np.asarray(predcov, dtype=np.float32)
    scans = np.asarray(scans, dtype=np.int64)
    obj = np.concatenate(
        [
            np.zeros((len(predcov), 1), dtype=np.float32),
            predcov - np.float32(penalty) * scans.astype(np.float32),
        ],
        axis=1,
    )
    return np.maximum(obj.argmax(axis=1), 1).astype(np.int32)


class ACRAdaptive:
    """Rank cells by coverage density, then pick a per-query prefix length."""

    def __init__(self, model, train_tau, deployment_tau, cell_sizes, costs=None):
        self.model = model
        self.prototypes = model.prototypes_normalized().detach().contiguous()
        self.train_tau = train_tau
        self.deployment_tau = deployment_tau
        self.log_density = torch.from_numpy(
            np.log(np.maximum(cell_sizes / cell_sizes.sum(), 1e-30)).astype(np.float32)
        )
        self.cell_sizes = torch.from_numpy(cell_sizes.astype(np.float32))
        self.costs = (
            None if costs is None else torch.as_tensor(costs, dtype=torch.float32)
        )
        if self.costs is not None and (
            self.costs.shape != self.cell_sizes.shape
            or not torch.isfinite(self.costs).all()
            or not torch.all(self.costs > 0)
        ):
            raise ValueError("Expected positive finite per-cell costs")
        self.lambda_ = None

    @torch.no_grad()
    def features(self, queries):
        out = (self.model.encode_query(queries) @ self.prototypes.T) / self.train_tau
        scaled = out * (self.train_tau / self.deployment_tau)
        density = scaled - self.log_density.view(1, -1)
        dvals, cells = torch.sort(density, dim=1, descending=True)
        selected_scaled = scaled.gather(1, cells)
        probs = torch.exp(
            selected_scaled - torch.logsumexp(scaled, dim=1, keepdim=True)
        )
        predcov = torch.cumsum(probs, dim=1)
        scans = torch.cumsum(self.cell_sizes[cells], dim=1)
        return (
            cells.cpu().numpy().astype(np.int32),
            dvals.cpu().numpy().astype(np.float32),
            predcov.cpu().numpy().astype(np.float32),
            scans.cpu().numpy().astype(np.int64),
        )

    def online(self, queries):
        if self.costs is not None:
            from .algorithm import choose_prefix

            logits = (
                self.model.encode_query(queries) @ self.prototypes.T
            ) / self.deployment_tau
            cells, budget = choose_prefix(
                logits, self.costs, self.lambda_, partial=False
            )
            distances = logits.gather(1, cells)
            return (
                cells.detach().numpy(),
                distances.detach().numpy(),
                budget.numpy(),
                None,
            )
        cells, dists, predcov, scans = self.features(queries)
        budget = prefix_budget(predcov, scans, self.lambda_)
        return cells, dists, budget, None
