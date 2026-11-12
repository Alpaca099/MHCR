"""Training-only scan measurements and budget derivation."""

import time
import resource
import numpy as np
import torch
from .algorithm import fit_cost


def measure(index, queries, sizes, probes, rerank=5000, repeats=3, raw_path=None):
    queries = np.ascontiguousarray(queries, np.float32)
    probes = sorted(set(min(index.nlist, int(p)) for p in probes))
    if len(queries) < 12 or repeats < 2 or min(probes) < 1 or rerank < 100:
        raise ValueError(
            "Cost fitting needs >=12 training queries, >=2 repeats and valid scan budgets"
        )
    centers = index.quantizer.reconstruct_n(0, index.nlist)
    order = np.argsort(-(queries @ centers.T), axis=1, kind="stable")

    def scan(i, p):
        cells = np.ascontiguousarray(order[i, :p][None], np.int64)
        coarse = np.ascontiguousarray(queries[i : i + 1] @ centers[cells[0]].T)
        index.nprobe = p
        before = resource.getrusage(resource.RUSAGE_SELF)
        t = time.perf_counter_ns()
        index.search_preassigned(queries[i : i + 1], rerank, cells, coarse)
        ms = (time.perf_counter_ns() - t) / 1e6
        after = resource.getrusage(resource.RUSAGE_SELF)
        return dict(
            query_row=i,
            lists=p,
            blocks=int(((sizes[cells[0]] + 31) // 32).sum()),
            scan_ms=ms,
            major_faults=after.ru_majflt - before.ru_majflt,
            minor_faults=after.ru_minflt - before.ru_minflt,
        )

    for i in range(min(8, len(queries))):
        scan(i, max(probes))
    rows = []
    for rep in range(repeats):
        for i in np.random.default_rng(rep).permutation(len(queries)):
            for p in probes:
                rows.append(dict(scan(int(i), p), repeat=rep))
    if raw_path is not None:
        from .storage import save

        save(raw_path, rows)
    fit = [r for r in rows if r["query_row"] % 4 != 0]
    cost, model = fit_cost(
        [[r["lists"], r["blocks"], r["scan_ms"]] for r in fit], sizes
    )
    held = [r for r in rows if r["query_row"] % 4 == 0]
    actual = np.array([r["scan_ms"] for r in held])
    prediction = (
        np.array([[1, r["lists"], r["blocks"]] for r in held]) @ model["coefficients"]
    )
    model.update(
        heldout_train_mape=float(
            np.mean(abs(prediction - actual) / np.maximum(actual, 1e-9))
        ),
        training_only=True,
        all_faults_retained=True,
        warmup_queries=min(8, len(queries)),
        repeats=repeats,
        rerank=rerank,
    )
    if sum(model["coefficients"][1:]) <= 0:
        raise ValueError(
            "Scan measurements produced no positive marginal cost; inspect raw observations"
        )
    return cost, model, rows


def derive_budget(router, queries, cells, sizes, unit):
    weights = sizes / sizes.sum() if unit == "fraction" else router.cost.numpy()
    required = []
    for q, gt in zip(queries, cells):
        order = router.route(torch.from_numpy(q[None]), 0.0)
        if len(order) != len(sizes) or len(np.unique(order)) != len(sizes):
            raise ValueError("Budget derivation requires a full cell ordering")
        positions = np.empty(len(order), np.int64)
        positions[order] = np.arange(len(order))
        prefix = np.cumsum(weights[order])
        required.append(float(np.sort(prefix[positions[gt]])[98]))
    total = float(weights.sum())
    lo, hi = np.quantile(required, [0.4, 0.8])
    lo = float(np.clip(lo, 0.001 * total, 0.999 * total))
    hi = float(np.clip(hi, lo, 0.9999 * total))
    if hi - lo < 0.0001 * total:
        lo = max(0.001 * total, lo * 0.8)
        hi = min(0.9999 * total, max(hi, lo + 0.0001 * total))
    return dict(
        range=[lo, hi],
        unit=unit,
        training_only=True,
        quantiles=[0.4, 0.8],
        required_for_99_neighbors=required,
        definition="Cell coverage for training budgets; not end-to-end recall or a guarantee",
    )
