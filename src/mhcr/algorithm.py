"""Cost-aware MHCR training/routing primitives; no data or old weights imported.

Tail means worst within-batch missing neighbor mass under a soft scan budget,
not a differentiable P99 latency loss. All target neighbors contribute uniformly.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def budget_gate(density, sizes, budget, temperature=0.25):
    """Weighted soft scan constraint, with implicit threshold differentiation.

    sizes and budget must have the same units, e.g. fractions of full gallery.
    Zero-size cells are valid; total size and every query budget must be positive.
    """
    assert density.ndim == 2 and sizes.shape == (density.shape[1],)
    assert torch.isfinite(density).all() and torch.isfinite(sizes).all()
    assert torch.all(sizes >= 0) and sizes.sum() > 0 and temperature > 0
    budget = torch.as_tensor(
        budget, dtype=density.dtype, device=density.device
    ).reshape(-1, 1)
    assert len(budget) in (1, len(density)) and torch.isfinite(budget).all()
    assert torch.all(budget > 0) and torch.all(budget < sizes.sum())
    with torch.no_grad():
        lo = density.amin(1, keepdim=True) - 30 * temperature
        hi = density.amax(1, keepdim=True) + 30 * temperature
        for _ in range(32):
            mid = (lo + hi) * 0.5
            g = torch.sigmoid((density - mid) / temperature)
            above = (g * sizes).sum(1, keepdim=True) > budget
            lo = torch.where(above, mid, lo)
            hi = torch.where(above, hi, mid)
        threshold = (lo + hi) * 0.5
        g = torch.sigmoid((density - threshold) / temperature)
        w = sizes * g * (1 - g)
        denom = w.sum(1, keepdim=True)
        assert torch.all(
            denom > torch.finfo(density.dtype).tiny
        ), "Degenerate soft constraint"
    threshold = (
        threshold + (w * (density - density.detach())).sum(1, keepdim=True) / denom
    )
    return torch.sigmoid((density - threshold) / temperature)


def datacomp_gate(density, cost, budget):
    """Frozen DataComp student gate: 12 bisections, temperature .25."""
    with torch.no_grad():
        lo = density.amin(1, keepdim=True) - 6
        hi = density.amax(1, keepdim=True) + 6
        for _ in range(12):
            mid = (lo + hi) / 2
            g = torch.sigmoid((density - mid) / 0.25)
            above = (g * cost).sum(1, keepdim=True) > budget
            lo = torch.where(above, mid, lo)
            hi = torch.where(above, hi, mid)
        threshold = (lo + hi) / 2
        g = torch.sigmoid((density - threshold) / 0.25)
        weight = cost * g * (1 - g)
    threshold = threshold + (weight * (density - density.detach())).sum(
        1, keepdim=True
    ) / weight.sum(1, keepdim=True).clamp(min=1e-9)
    return torch.sigmoid((density - threshold) / 0.25)


def training_loss(
    logits,
    cells,
    cost,
    sizes,
    budget,
    family="budget_tail",
    teacher_prob=None,
    gate_variant="implicit32",
):
    assert cells.ndim == 2 and cells.shape == (len(logits), 100)
    assert torch.all((cells >= 0) & (cells < logits.shape[1]))
    assert cost.shape == sizes.shape == (logits.shape[1],) and torch.all(cost > 0)
    assert torch.isfinite(cost).all()
    logp = F.log_softmax(logits, 1)
    ce = -logp.gather(1, cells).mean()
    distill = logp.new_zeros(())
    if teacher_prob is not None:
        teacher_prob = teacher_prob.detach()
        assert (
            teacher_prob.shape == logp.shape
            and torch.all(teacher_prob >= 0)
            and torch.isfinite(teacher_prob).all()
        )
        assert torch.all(teacher_prob.sum(1) > 0)
        teacher_prob = teacher_prob / teacher_prob.sum(1, keepdim=True)
        distill = -(teacher_prob * logp).sum(1).mean()
    anchor = ce + 0.25 * distill
    if family == "ce":
        return anchor, {"ce": ce, "distill": distill}
    assert family in ("budget_mean", "budget_tail")
    if gate_variant not in ("implicit32", "datacomp12"):
        raise ValueError("Unknown gate implementation")
    g = (datacomp_gate if gate_variant == "datacomp12" else budget_gate)(
        logp - cost.log(), sizes, budget
    )
    miss = 1 - g.gather(1, cells).mean(1)
    primary = (
        miss.mean()
        if family == "budget_mean"
        else 0.5 * miss.mean()
        + 0.5 * torch.topk(miss, max(1, math.ceil(0.2 * len(miss)))).values.mean()
    )
    loss = 10 * primary + 0.1 * anchor
    return loss, {
        "ce": ce,
        "distill": distill,
        "primary": primary,
        "mean_missing": miss.mean(),
        "soft_scanned": (g * sizes).sum(1).mean(),
    }


def choose_prefix(logits, cost, penalty, partial=True):
    """All prototypes scored; conservative partial ranking with explicit tie fallback.

    Equal density is ordered by cell ID. Full-K is valid, including penalty zero.
    No fixed maximum probe count. Return selected ordering and per-query budget.
    """
    assert logits.ndim == 2 and cost.shape == (logits.shape[1],)
    assert (
        torch.isfinite(logits).all()
        and torch.isfinite(cost).all()
        and torch.all(cost > 0)
    )
    assert math.isfinite(penalty) and penalty >= 0
    p = torch.softmax(logits, 1)
    density = logits - cost.log()
    k = logits.shape[1]
    width = k
    if partial and penalty > 0:
        guard = 2 * np.finfo(np.float32).eps * torch.clamp(p.double().sum(1), min=1)
        possible = p.double() - penalty * cost.double() >= -guard[:, None]
        floor = torch.where(possible, density, torch.inf).amin(1)
        width = min(
            k,
            max(64, ((int((density >= floor[:, None]).sum(1).max()) + 63) // 64) * 64),
        )
    if width < k:
        values, cells = torch.topk(density, width, dim=1, sorted=True)
        if torch.any(values[:, 1:] == values[:, :-1]) or torch.any(
            (density == values[:, -1:]).sum(1) > 1
        ):
            cells = torch.argsort(density, dim=1, descending=True, stable=True)
    else:
        cells = torch.argsort(density, dim=1, descending=True, stable=True)
    coverage = torch.cumsum(p.gather(1, cells), 1).double()
    total = torch.cumsum(cost.double()[cells], 1)
    obj = torch.cat(
        [
            torch.zeros((len(logits), 1), dtype=torch.float64),
            coverage - penalty * total,
        ],
        1,
    )
    budget = obj.argmax(1).clamp(min=1)
    if penalty == 0:
        budget[:] = k  # Explicit full-coverage endpoint, including softmax underflow.
    return cells, budget


class Router:
    def __init__(self, model, cost, quantized=True):
        self.mean = model.query_mean.detach()
        self.encoder = model.encoder.eval()
        prototypes = model.prototypes_normalized().detach().contiguous()
        linear = nn.Linear(prototypes.shape[1], len(prototypes), bias=False)
        linear.weight = nn.Parameter(prototypes.clone(), requires_grad=False)
        self.scorer = nn.Sequential(linear).eval()
        self.tau = model.tau
        if quantized:
            self.encoder = torch.ao.quantization.quantize_dynamic(
                self.encoder, {nn.Linear}, dtype=torch.qint8, inplace=False
            )
            self.scorer = torch.ao.quantization.quantize_dynamic(
                self.scorer, {nn.Linear}, dtype=torch.qint8, inplace=False
            )
        self.cost = torch.as_tensor(cost, dtype=torch.float32)
        self.quantized = quantized

    @torch.inference_mode()
    def route(self, features, penalty, partial=True):
        assert len(features) == 1, "Latency protocol is single query, single CPU thread"
        logits = (
            self.scorer(F.normalize(self.encoder(features - self.mean), dim=-1))
            / self.tau
        )
        cells, budget = choose_prefix(logits, self.cost, penalty, partial)
        return np.ascontiguousarray(cells[0, : int(budget[0])].numpy(), np.int64)


def fit_cost(observations, sizes, block=32):
    """Nonnegative least squares for measured scan-only time, intercept kept separate.

    Each row is (number of lists, total padded PQ blocks, measured milliseconds).
    Enumerate the 8 active sets for three coefficients; never use target latency.
    """
    a = np.asarray(observations, np.float64)
    sizes = np.asarray(sizes, np.int64)
    assert a.ndim == 2 and a.shape[1] == 3 and len(a) >= 6 and np.isfinite(a).all()
    assert np.all(a[:, :2] > 0) and np.all(a[:, 2] >= 0) and np.all(sizes >= 0)
    X = np.column_stack([np.ones(len(a)), a[:, :2]])
    scale = np.linalg.norm(X, axis=0)
    Z = X / scale
    assert (
        np.linalg.matrix_rank(Z) == 3
    ), "Cost observations do not distinguish list and scan work"
    best = None
    for mask in range(8):
        columns = [i for i in range(3) if mask & (1 << i)]
        coef = np.zeros(3)
        if columns:
            v = np.linalg.lstsq(Z[:, columns], a[:, 2], rcond=None)[0]
            if np.any(v < 0):
                continue
            coef[columns] = v
        error = float(np.square(Z @ coef - a[:, 2]).sum())
        if best is None or error < best[0]:
            best = (error, coef / scale)
    coef = best[1]
    cost = np.maximum(1e-7, coef[1] + coef[2] * ((sizes + block - 1) // block)).astype(
        np.float32
    )
    return cost, dict(
        coefficients=coef.tolist(),
        columns=["fixed_intercept_ms", "list_ms", "block32_ms"],
        rmse_ms=float(np.sqrt(best[0] / len(a))),
        intercept_excluded_from_marginal=True,
        scope="Empirical scan-only cost proxy; end-to-end latency must still be measured",
    )
